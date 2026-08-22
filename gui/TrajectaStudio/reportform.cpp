#include "reportform.h"

#include "confirmdialog.h"
#include "framelessdialog.h"
#include "systeminfo.h"

#include <QBuffer>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QHttpMultiPart>
#include <QHttpPart>
#include <QIcon>
#include <QImage>
#include <QImageReader>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPixmap>
#include <QPainter>
#include <QTimer>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSysInfo>
#include <QTextStream>
#include <QThread>
#include <QUrl>
#include <QVBoxLayout>
#include <QVector>

namespace TrajectaUi {

namespace {

// ---------------------------------------------------------------------------
// Where a report goes
// ---------------------------------------------------------------------------
// To a relay, not to a mailbox. The relay keeps the destination address in its
// own configuration and forwards to it, so what ships inside this binary is
// only the endpoint and its public key. The project's address is therefore
// nowhere in the program and cannot be lifted out of it with `strings` — which
// is the whole reason for going through a relay rather than speaking SMTP from
// here. An account able to send mail would need credentials, and a credential
// inside a program anyone can download is a published credential.
//
// Should either constant below ever be blank, the form still opens and still
// explains itself, but Send is disabled and says why. A button that admits it is
// not connected is better than one that accepts a report and drops it.
//
// The request is multipart/form-data rather than JSON, because a report can
// carry images. One shape for both cases rather than two code paths: the relay
// reads it with a single `await request.formData()`.
//
// The relay's own address, deliberately not the mailbox it forwards to. Worth
// moving onto a domain under our own control before long: this URL is frozen
// into every installer ever released, so the day the backend moves, it is what
// every copy already in the world goes on calling.
constexpr auto kEndpoint = "https://trajectastudio.netlify.app/report";
// Public by construction — it travels inside a program anyone can download and
// read. It exists so the endpoint is not used by a passing scanner, not to
// prove who is calling; the checks that matter are the relay's own.
constexpr auto kAccessKey = "ebf9a4491f2259efd134bb79924f0513";
constexpr auto kSubject = "TRAJECTA REPORT";

constexpr int kDialogWidth = 600;
constexpr int kDialogHeight = 660;
constexpr int kButtonWidth = 92;
// Enough for a long account of what went wrong, short of enough to be worth
// abusing the form with.
constexpr int kMaxMessage = 8000;

// --- images ---------------------------------------------------------------
// PNG and JPEG, and deliberately nothing else. They are what a screenshot and
// a photograph actually are, every mail client in the world previews both, and
// both are read by Qt itself without depending on an image plugin that a given
// installation may not carry.
constexpr int kMaxImages = 4;
// Long edge. 1920 leaves a full-HD screenshot untouched, which is the common
// case and the one where every pixel of text matters; a 4K screenshot is halved
// and stays perfectly readable.
constexpr int kMaxImageEdge = 1920;
constexpr int kJpegQuality = 85;
// Above this an encoded PNG is almost certainly a photograph rather than a
// screenshot, and JPEG will do the same job several times smaller.
constexpr int kPngPreferredMax = 1500 * 1024;
// The whole report, images included.
//
// Set by the tightest ceiling in the chain, which is not the mail system: a
// serverless function is handed its request whole and refuses anything much
// over 6 MB, and the body is base64-encoded on the way in, which costs a third
// again. Four megabytes of request therefore arrives as about five and a third
// and fits, while four screenshots at a megabyte each is more than anyone
// attaches to a bug report.
constexpr qint64 kMaxTotalBytes = 4 * 1024 * 1024;
constexpr int kThumbEdge = 64;

struct Attachment {
    QString name;       // what the far end will see it called
    QByteArray data;    // already re-encoded, ready to send
    QByteArray format;  // "png" or "jpeg"
    QPixmap thumb;
};

// Qt names a .jpg file's format "jpeg", so that is the spelling compared
// against; the extensions offered in the file dialog are a separate list.
bool isAcceptedFormat(const QByteArray &format)
{
    const QByteArray f = format.toLower();
    return f == "png" || f == "jpeg" || f == "jpg";
}

QString acceptedFormatsSentence()
{
    return QObject::tr("Only PNG and JPEG images can be attached "
                       "(.png, .jpg, .jpeg).");
}

QString imageFilter()
{
    return QObject::tr("Images (*.png *.jpg *.jpeg);;All files (*)");
}

QString humanSize(qint64 bytes)
{
    if (bytes >= 1024 * 1024)
        return QStringLiteral("%1 MB").arg(QString::number(bytes / (1024.0 * 1024.0), 'f', 1));
    return QStringLiteral("%1 kB").arg(qMax<qint64>(1, bytes / 1024));
}

// Reads an image, shrinks it if it is bigger than anyone needs, and re-encodes
// it for sending.
//
// Re-encoding rather than attaching the file as it is, for three reasons: a
// phone photograph is several megabytes of detail nobody will look at, an
// unexpected format would arrive at the far end unopenable, and a file straight
// off disk carries metadata — including, on a photograph, the place it was
// taken. What leaves here is pixels and nothing else.
//
// `error` is filled with a sentence fit to show the user, naming the file.
bool prepareAttachment(const QString &path, Attachment &out, QString *error)
{
    const QString shown = QFileInfo(path).fileName();

    // What the file *is*, not what it is called. A screenshot renamed to .png
    // is still read correctly, and a PDF renamed to .png is refused with the
    // right reason instead of failing later as an unreadable image.
    const QByteArray sniffed = QImageReader::imageFormat(path);
    if (sniffed.isEmpty()) {
        if (error) {
            *error = QObject::tr("%1 is not an image Trajecta can read. %2")
                         .arg(shown, acceptedFormatsSentence());
        }
        return false;
    }
    if (!isAcceptedFormat(sniffed)) {
        if (error) {
            *error = QObject::tr("%1 is a %2 image. %3")
                         .arg(shown, QString::fromLatin1(sniffed).toUpper(),
                              acceptedFormatsSentence());
        }
        return false;
    }

    QImageReader reader(path);
    reader.setAutoTransform(true);   // honour the orientation tag of a photo
    QImage img = reader.read();
    if (img.isNull()) {
        if (error) {
            *error = QObject::tr("%1 could not be read (%2).")
                         .arg(shown, reader.errorString());
        }
        return false;
    }
    if (img.width() > kMaxImageEdge || img.height() > kMaxImageEdge) {
        img = img.scaled(kMaxImageEdge, kMaxImageEdge, Qt::KeepAspectRatio,
                         Qt::SmoothTransformation);
    }

    // PNG first: it is lossless, which is what a screenshot of text deserves.
    // If the result is large the source was a photograph in all but name, and
    // JPEG is used instead.
    QByteArray encoded;
    QByteArray format = "png";
    {
        QBuffer buf(&encoded);
        buf.open(QIODevice::WriteOnly);
        if (!img.save(&buf, "PNG")) {
            if (error)
                *error = QObject::tr("%1 could not be encoded for sending.").arg(shown);
            return false;
        }
    }
    if (encoded.size() > kPngPreferredMax) {
        QByteArray asJpeg;
        QBuffer buf(&asJpeg);
        buf.open(QIODevice::WriteOnly);
        if (img.save(&buf, "JPEG", kJpegQuality) && asJpeg.size() < encoded.size()) {
            encoded = asJpeg;
            format = "jpeg";
        }
    }

    out.name = QFileInfo(path).completeBaseName() + QLatin1Char('.')
               + QString::fromLatin1(format);
    out.data = encoded;
    out.format = format;
    out.thumb = QPixmap::fromImage(
        img.scaled(kThumbEdge, kThumbEdge, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    return true;
}

// What the report carries besides the text: the version that produced it and
// the machine it ran on. "It crashed" is not a report anyone can act on, and
// asking the user to look these up by hand is asking them not to bother.
//
// Shown in the dialog in plain sight, not gathered behind their back. Nothing
// here identifies a person or a machine — no name, no path, no serial.
QString diagnostics()
{
    const double ramGb = SystemInfo::totalRamMb() / 1024.0;
    QString s = QStringLiteral("Trajecta Studio %1")
                    .arg(QCoreApplication::applicationVersion());
    s += QStringLiteral(" · %1").arg(QSysInfo::prettyProductName());
    s += QStringLiteral(" · %1").arg(QSysInfo::currentCpuArchitecture());
    s += QStringLiteral(" · %1 threads").arg(QThread::idealThreadCount());
    s += QStringLiteral(" · %1 GB RAM").arg(QString::number(ramGb, 'f', 1));
    return s;
}

// The last resort when the message cannot be sent: the user has just written
// several paragraphs and must not lose them because a firewall said no. The
// images are written beside the text, so a report saved now can still be sent
// by hand later.
void offerToSave(QWidget *parent, const QString &bodyText, const QString &replyTo,
                 const QVector<Attachment> &images)
{
    if (!confirm(parent, QObject::tr("Keep a copy?"),
                 QObject::tr("The report could not be sent.\n\n"
                             "Save what you wrote, so it is not lost?"),
                 QObject::tr("Save..."), QObject::tr("Discard"), 20,
                 Fill::Accept)) {
        return;
    }
    const QString stamp =
        QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmm"));
    const QString path = QFileDialog::getSaveFileName(
        parent, QObject::tr("Save the report"),
        QStringLiteral("trajecta-report-%1.txt").arg(stamp),
        QObject::tr("Text files (*.txt);;All files (*)"));
    if (path.isEmpty())
        return;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        notify(parent, QObject::tr("Could not save"),
               QObject::tr("The file could not be written:\n%1").arg(path));
        return;
    }
    QTextStream out(&f);
    out << QLatin1String(kSubject) << "\n\n" << diagnostics() << "\n";
    if (!replyTo.isEmpty())
        out << QObject::tr("Reply to: ") << replyTo << "\n";
    out << "\n" << bodyText << "\n";
    f.close();

    // The images beside the text file and named after it, so the set stays
    // together and it is obvious which report they belong to.
    const QFileInfo info(path);
    QStringList failed;
    for (int i = 0; i < images.size(); ++i) {
        const QString imgPath =
            info.dir().filePath(QStringLiteral("%1-%2.%3")
                                    .arg(info.completeBaseName())
                                    .arg(i + 1)
                                    .arg(QString::fromLatin1(images.at(i).format)));
        QFile img(imgPath);
        if (img.open(QIODevice::WriteOnly))
            img.write(images.at(i).data);
        else
            failed << QFileInfo(imgPath).fileName();
    }
    if (!failed.isEmpty()) {
        notify(parent, QObject::tr("Some images were not saved"),
               QObject::tr("The text was saved, but these could not be written:\n%1")
                   .arg(failed.join(QStringLiteral(", "))));
    }
}

} // namespace

void showReportForm(QWidget *parent, const QString &selfTestMessage)
{
    // The window, not the widget that asked: a dialog parented inside a card
    // inherits the card's "clear the background of everything in here" rule and
    // comes up black. See dialogHost() in confirmdialog.cpp.
    FramelessDialog dialog(parent ? parent->window() : nullptr,
                           QObject::tr("Report a problem"));
    dialog.setObjectName(QStringLiteral("ConfirmDialog"));
    dialog.setFixedSize(kDialogWidth, FramelessDialog::kTitleBarHeight + kDialogHeight);

    auto *layout = dialog.contentLayout();
    layout->setContentsMargins(28, 24, 28, 22);
    layout->setSpacing(10);

    auto *intro = new QLabel(
        QObject::tr("Describe the problem, or the improvement you would like to "
                    "see. If it is a bug, what you were doing when it happened "
                    "is the most useful thing you can write."),
        &dialog);
    intro->setObjectName(QStringLiteral("ConfirmText"));
    intro->setWordWrap(true);
    // Ranged left, unlike the confirmations: this is a form to fill in, and a
    // centred instruction above a left-aligned field reads as decoration.
    intro->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    layout->addWidget(intro);

    auto *body = new QPlainTextEdit(&dialog);
    body->setObjectName(QStringLiteral("ReportBody"));
    body->setPlaceholderText(QObject::tr("What happened, and what you expected "
                                         "to happen instead."));
    body->setTabChangesFocus(true);   // Tab moves on; nobody indents a report
    layout->addWidget(body, 1);

    // --- images ---
    auto *imageHeader = new QHBoxLayout;
    imageHeader->setSpacing(8);
    auto *imageLabel = new QLabel(&dialog);
    imageLabel->setObjectName(QStringLiteral("ReportMeta"));
    auto *addImages = new QPushButton(QObject::tr("Add..."), &dialog);
    auto *removeImage = new QPushButton(QObject::tr("Remove"), &dialog);
    for (QPushButton *b : { addImages, removeImage }) {
        b->setObjectName(QStringLiteral("ConfirmButton"));
        b->setCursor(Qt::PointingHandCursor);
        b->setAutoDefault(false);
    }
    imageHeader->addWidget(imageLabel);
    imageHeader->addStretch(1);
    imageHeader->addWidget(addImages);
    imageHeader->addWidget(removeImage);
    layout->addLayout(imageHeader);

    auto *shots = new QListWidget(&dialog);
    shots->setObjectName(QStringLiteral("ReportShots"));
    // A strip of thumbnails rather than a list of file names: the whole point of
    // attaching a screenshot is that it is the *right* screenshot, and only the
    // picture itself says so.
    shots->setViewMode(QListView::IconMode);
    shots->setIconSize(QSize(kThumbEdge, kThumbEdge));
    shots->setFlow(QListView::LeftToRight);
    shots->setWrapping(false);
    shots->setMovement(QListView::Static);
    shots->setSelectionMode(QAbstractItemView::SingleSelection);
    shots->setFixedHeight(kThumbEdge + 30);
    shots->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    shots->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    layout->addWidget(shots);

    auto *replyRow = new QHBoxLayout;
    replyRow->setSpacing(8);
    auto *replyLabel = new QLabel(QObject::tr("Your email (optional):"), &dialog);
    replyLabel->setObjectName(QStringLiteral("ReportMeta"));
    auto *replyEdit = new QLineEdit(&dialog);
    replyEdit->setPlaceholderText(QObject::tr("only needed if you want an answer"));
    replyEdit->setMaxLength(200);
    replyRow->addWidget(replyLabel);
    replyRow->addWidget(replyEdit, 1);
    layout->addLayout(replyRow);

    // Sent with the report, and shown here because it is: the user can read
    // exactly what is attached before pressing Send.
    auto *meta = new QLabel(QObject::tr("Sent with your report: %1").arg(diagnostics()),
                            &dialog);
    meta->setObjectName(QStringLiteral("ReportMeta"));
    meta->setWordWrap(true);
    layout->addWidget(meta);

    auto *status = new QLabel(&dialog);
    status->setObjectName(QStringLiteral("ReportStatus"));
    status->setWordWrap(true);
    layout->addWidget(status);

    auto *buttons = new QHBoxLayout;
    buttons->setSpacing(12);
    buttons->addStretch(1);
    auto *cancel = new QPushButton(QObject::tr("Cancel"), &dialog);
    // Red, because pressing it throws away what has been typed — but built as a
    // #ConfirmButton wearing a danger fill rather than as a #DangerButton. That
    // rule carries no radius and no padding of its own, so a button using it
    // came out a different shape and size from the Send button next to it. The
    // property mechanism changes the colour and nothing else.
    cancel->setObjectName(QStringLiteral("ConfirmButton"));
    cancel->setProperty("fill", QStringLiteral("danger"));
    auto *send = new QPushButton(QObject::tr("Send"), &dialog);
    send->setObjectName(QStringLiteral("ConfirmButton"));
    // Filled in the theme's accent, by the same property the confirmations use.
    send->setProperty("fill", QStringLiteral("accent"));
    for (QPushButton *b : { cancel, send }) {
        b->setMinimumWidth(kButtonWidth);
        b->setCursor(Qt::PointingHandCursor);
        b->setAutoDefault(false);
        buttons->addWidget(b);
    }
    layout->addLayout(buttons);

    QVector<Attachment> images;
    const bool configured = *kEndpoint != '\0' && *kAccessKey != '\0';

    // How much has been added and how much room is left. Shown rather than
    // enforced silently: a refusal with no number attached leaves the user
    // guessing which image was the problem.
    const auto refreshImageLine = [&] {
        qint64 total = 0;
        for (const Attachment &a : images)
            total += a.data.size();
        if (images.isEmpty()) {
            imageLabel->setText(QObject::tr("Images (optional) — PNG or JPEG:"));
        } else {
            imageLabel->setText(QObject::tr("Images: %1 of %2, %3")
                                    .arg(images.size())
                                    .arg(kMaxImages)
                                    .arg(humanSize(total)));
        }
        // No strip until there is something in it. An empty box the height of a
        // thumbnail is a large piece of the dialog spent saying nothing, and the
        // writing area — which has the stretch — takes the room instead.
        shots->setVisible(!images.isEmpty());
        addImages->setEnabled(images.size() < kMaxImages);
        removeImage->setEnabled(shots->currentRow() >= 0);
    };

    const auto refreshSend = [&] {
        const int n = body->toPlainText().trimmed().size();
        if (!configured) {
            status->setText(QObject::tr("Reporting is not available in this build."));
            send->setEnabled(false);
            return;
        }
        send->setEnabled(n > 0 && n <= kMaxMessage);
        if (n > kMaxMessage) {
            status->setText(QObject::tr("Too long by %1 characters.")
                                .arg(n - kMaxMessage));
        } else {
            status->clear();
        }
    };

    QObject::connect(body, &QPlainTextEdit::textChanged, &dialog, refreshSend);
    QObject::connect(shots, &QListWidget::currentRowChanged, &dialog,
                     [removeImage](int row) { removeImage->setEnabled(row >= 0); });

    // Taking files and turning them into attachments, kept out of the button's
    // handler so the self-test below goes in by the same door the user does.
    const auto addPaths = [&](const QStringList &picked) {
        if (picked.isEmpty())
            return;
        qint64 total = 0;
        for (const Attachment &a : images)
            total += a.data.size();

        // Every refusal is collected and shown once at the end, rather than a
        // box per file: choosing five things and being told off five times is
        // its own small punishment.
        QStringList refused;
        for (const QString &path : picked) {
            if (images.size() >= kMaxImages) {
                refused << QObject::tr("%1 — no room for more than %2 images.")
                               .arg(QFileInfo(path).fileName())
                               .arg(kMaxImages);
                continue;
            }
            Attachment a;
            QString error;
            if (!prepareAttachment(path, a, &error)) {
                refused << error;
                continue;
            }
            if (total + a.data.size() > kMaxTotalBytes) {
                refused << QObject::tr("%1 (%2) — the report would pass its %3 limit.")
                               .arg(QFileInfo(path).fileName(),
                                    humanSize(a.data.size()),
                                    humanSize(kMaxTotalBytes));
                continue;
            }
            total += a.data.size();
            images.append(a);
            auto *item = new QListWidgetItem(QIcon(a.thumb), QString(), shots);
            item->setToolTip(QStringLiteral("%1 — %2")
                                 .arg(QFileInfo(path).fileName(), humanSize(a.data.size())));
        }
        refreshImageLine();
        if (!refused.isEmpty()) {
            notify(&dialog,
                   refused.size() == 1 ? QObject::tr("This image was not added")
                                       : QObject::tr("Some images were not added"),
                   refused.join(QStringLiteral("\n\n")));
        }
    };

    QObject::connect(addImages, &QPushButton::clicked, &dialog, [&] {
        addPaths(QFileDialog::getOpenFileNames(
            &dialog, QObject::tr("Add images to the report"), QString(), imageFilter()));
    });

    QObject::connect(removeImage, &QPushButton::clicked, &dialog, [&] {
        const int row = shots->currentRow();
        if (row < 0 || row >= images.size())
            return;
        images.remove(row);
        delete shots->takeItem(row);
        refreshImageLine();
    });

    refreshImageLine();
    refreshSend();

    QNetworkAccessManager net;

    QObject::connect(cancel, &QPushButton::clicked, &dialog, &QDialog::reject);
    QObject::connect(send, &QPushButton::clicked, &dialog, [&] {
        const QString text = body->toPlainText().trimmed();
        if (text.isEmpty())
            return;
        const QString replyTo = replyEdit->text().trimmed();

        // multipart/form-data: the text fields as form fields, each image as a
        // file part. Qt writes the Content-Type header itself, boundary and all,
        // so it must not be set here.
        auto *multi = new QHttpMultiPart(QHttpMultiPart::FormDataType);
        const auto addField = [multi](const QString &name, const QString &value) {
            QHttpPart part;
            part.setHeader(QNetworkRequest::ContentDispositionHeader,
                           QVariant(QStringLiteral("form-data; name=\"%1\"").arg(name)));
            part.setBody(value.toUtf8());
            multi->append(part);
        };
        addField(QStringLiteral("access_key"), QString::fromLatin1(kAccessKey));
        addField(QStringLiteral("subject"), QString::fromLatin1(kSubject));
        addField(QStringLiteral("from_name"), QStringLiteral("Trajecta Studio"));
        // The diagnostics travel inside the message rather than as fields of
        // their own: whoever reads the mail gets one thing to read.
        addField(QStringLiteral("message"),
                 diagnostics() + QStringLiteral("\n\n") + text);
        if (!replyTo.isEmpty())
            addField(QStringLiteral("replyto"), replyTo);

        for (int i = 0; i < images.size(); ++i) {
            const Attachment &a = images.at(i);
            QHttpPart part;
            part.setHeader(QNetworkRequest::ContentTypeHeader,
                           QVariant(QStringLiteral("image/%1")
                                        .arg(QString::fromLatin1(a.format))));
            part.setHeader(
                QNetworkRequest::ContentDispositionHeader,
                QVariant(QStringLiteral("form-data; name=\"attachment%1\"; filename=\"%2\"")
                             .arg(i + 1)
                             .arg(a.name)));
            part.setBody(a.data);
            multi->append(part);
        }

        // Braces, and the URL named rather than built inside the argument list.
        // With parentheses both lines are read as function declarations —
        // QLatin1String(kEndpoint) looks like a parameter named kEndpoint — and
        // the errors that follow talk about conversions that have nothing to do
        // with it.
        const QUrl endpoint{QLatin1String(kEndpoint)};
        QNetworkRequest request{endpoint};
        request.setRawHeader("Accept", "application/json");

        send->setEnabled(false);
        cancel->setEnabled(false);
        addImages->setEnabled(false);
        removeImage->setEnabled(false);
        status->setText(QObject::tr("Sending..."));

        QNetworkReply *reply = net.post(request, multi);
        // The multipart has to outlive the call and die with the reply, which is
        // what parenting it to the reply arranges.
        multi->setParent(reply);

        // Bound to the dialog, so the connection dies with it: the lambda holds
        // references to locals of this function, and must never run after they
        // have gone.
        QObject::connect(reply, &QNetworkReply::finished, &dialog, [&, reply] {
            reply->deleteLater();
            const QByteArray answer = reply->readAll();
            bool ok = reply->error() == QNetworkReply::NoError;
            if (ok) {
                // A relay can answer 200 and still have refused the form — a
                // wrong key, or the monthly allowance spent — so the status code
                // alone is not the answer. A reply that is not JSON at all, or
                // that says nothing about success, is taken at the word of its
                // status code.
                const QJsonObject o = QJsonDocument::fromJson(answer).object();
                if (o.contains(QStringLiteral("success")))
                    ok = o.value(QStringLiteral("success")).toBool(false);
            }
            if (ok) {
                notify(&dialog, QObject::tr("Report sent"),
                       QObject::tr("Thank you. Your report has been sent."));
                dialog.accept();
                return;
            }
            send->setEnabled(true);
            cancel->setEnabled(true);
            refreshImageLine();
            status->setText(QObject::tr("Could not send: %1")
                                .arg(reply->error() == QNetworkReply::NoError
                                         ? QObject::tr("the report was refused")
                                         : reply->errorString()));
            offerToSave(&dialog, text, replyTo, images);
        });
    });

    // The self-test: fill in, attach, send, all without a hand on the mouse.
    // The image is drawn here rather than read from somewhere on the machine, so
    // the test carries the same thing wherever it runs — and it goes through
    // addPaths() like any other file, which means prepareAttachment() is under
    // test too, not bypassed.
    if (!selfTestMessage.isEmpty()) {
        body->setPlainText(selfTestMessage);
        QImage probe(480, 300, QImage::Format_RGB32);
        probe.fill(QColor(28, 34, 42));
        {
            QPainter p(&probe);
            p.setPen(QColor(126, 168, 160));
            QFont f = p.font();
            f.setPixelSize(22);
            p.setFont(f);
            p.drawText(QRect(0, 0, 480, 300), Qt::AlignCenter,
                       QStringLiteral("Trajecta\nreport self-test\n%1")
                           .arg(QDateTime::currentDateTime().toString(Qt::ISODate)));
        }
        const QString probePath =
            QDir(QDir::tempPath()).filePath(QStringLiteral("trajecta-selftest.png"));
        if (probe.save(probePath, "PNG"))
            addPaths({ probePath });
        // After the dialog is up, so the click lands on a laid-out button.
        QTimer::singleShot(400, &dialog, [send] {
            if (send->isEnabled())
                send->click();
        });
    }

    centreOnScreen(dialog, parent);
    dialog.exec();
}

} // namespace TrajectaUi
