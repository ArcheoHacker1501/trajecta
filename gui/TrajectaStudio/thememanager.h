#pragma once

#include <QColor>
#include <QString>
#include <QVector>

// ThemeManager — the colour palettes offered by the gear menu.
//
// theme.qss is written in the default dark palette and stays the single source
// of truth for the *shape* of the UI (radii, paddings, which element gets which
// role). A palette is therefore not a second stylesheet but a colour map: the
// literal values of the dark theme rewritten to another set. Applying the dark
// palette is the identity, so it is guaranteed to look exactly as it always
// has, and a new palette can never accidentally change a layout.
//
// Roles that must keep their meaning (running amber, failed red, the satellite
// canvas) are mapped explicitly by each palette rather than derived, so a light
// theme can lighten its own backgrounds without washing out the status chips.
class ThemeManager
{
public:
    struct Theme {
        QString id;        // stored in QSettings
        QString name;      // shown in the menu
        bool light = false;
        // Colours the Viewer paints itself, outside the stylesheet. (The map
        // canvas is not one of them any more: it takes the card colour, so it
        // follows the panels on every other page.)
        QColor overlayPen;
        // Dark-hex -> theme-hex. Empty for the default dark palette.
        QVector<QPair<QString, QString>> map;
        // Appended after the colour-mapped sheet, so a palette can also change
        // shape — radii, weights, letter spacing, type. Only the themes that
        // need it set this; the colour map alone cannot express a "style".
        QString extraQss;
        // The map canvas is paper over the artwork rather than a solid slab.
        // Only meaningful on a theme that has something behind it to show, and
        // it is not free: a canvas that does not paint its own background makes
        // every repaint go up to the window's picture first. Set it for the
        // themes where the look is worth that, and nowhere else.
        bool translucentCanvas = false;
    };

    static const QVector<Theme> &themes();
    static int indexOfId(const QString &id);   // -1 when unknown

    // Theme used when the user has never chosen one. Deliberately not index 0:
    // that slot has to stay the identity colour map (see themes()), which is a
    // constraint of how palettes are built, not a statement about which theme
    // should greet a new user.
    static int defaultIndex();

    // --- UI font -----------------------------------------------------------
    // Chosen independently of the palette. Index 0 means "whatever the theme
    // asks for", which is how Washi keeps its serif; any other choice is
    // appended after the theme's own rules and therefore overrides them.
    struct FontChoice {
        QString id;       // stored in QSettings
        QString name;     // shown in the menu
        QString family;   // empty for "theme default"
    };
    static const QVector<FontChoice> &fonts();
    static int currentFont();
    static void setFont(int index);   // stores and re-applies the active theme

    // Reads :/theme.qss, applies the palette's colour map and installs the
    // result on qApp, together with a matching QPalette (Fusion draws combo
    // popups, menus and message boxes from the palette, not the stylesheet).
    static void apply(int index);

    // Index currently applied. Defaults to the dark palette.
    static int current();

    static const Theme &theme(int index);

    // Translates one of the dark palette's colours into the active theme.
    // For the handful of places that paint outside the stylesheet — the
    // console, the map canvas, the status line, the Guide's own <style> block.
    static QColor mapped(const char *darkHex);
    static bool isLight();

    // Corner radius of QFrame#Card under the active theme, in device-independent
    // pixels. The walkthrough needs it: a lit card has to be cut out in the
    // card's own shape, and two themes restate that radius in their shape
    // overrides. Mirrors theme.qss and the extraQss blocks in thememanager.cpp
    // — the one number in this file that is written down twice, because QSS
    // cannot be queried back.
    static int cardRadius();
};
