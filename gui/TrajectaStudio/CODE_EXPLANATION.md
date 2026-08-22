# Trajecta Studio — Complete explanation (general logic and code)

This document explains **how the Trajecta graphical interface works**, first
at the architecture level and then diving into the code file by file,
block by block.

Parts 1 to 3 describe the interface as it was first built. **Part 4** covers the
code added since — the top bar, batch processing, the map viewer, checkpointing
and the guided tour — and is the current description wherever the two disagree.
**Part 5** leaves the interface entirely and explains the propagation algorithm
in the engine, because every setting in the form is an argument to it.

---

## Part 1 — The general logic

### 1.1 The starting problem

`trajecta.exe` is an **interactive console program**: it does not accept
command-line arguments. It asks a series of questions one at a time
(`Choose computation mode:`, `Enter path to DEM file (.tif):`, …) and reads
the answers typed by the user. The sequence of questions is not fixed: it
depends on the answers (e.g. answering "no" to cost modifiers skips 4
questions; a failed validation makes some questions repeat).

### 1.2 The architectural choice: drive the engine, don't rewrite it

Trajecta Studio does **not** re-implement any computation. It launches the
*very same* `trajecta.exe` as a child process and "types" the answers on the
user's behalf. Advantages:

1. **Zero divergence**: GUI and console use the same engine; every bugfix or
   new feature of the engine is automatically available in the GUI.
2. **Total parameter coverage**: every question of the interactive flow has a
   corresponding widget in the form (requirement 1).
3. **Transparency**: the live console on the Run page shows exactly what the
   engine prints, as if the user were running it from PowerShell.

### 1.3 Why the mechanism is reliable (the key detail)

Driving an interactive program through pipes is notoriously fragile for one
reason: when stdout is not a console but a pipe, C++ buffers it, and the
questions might never reach the GUI before the engine blocks waiting for the
answer (deadlock). Here, however, there is a language guarantee:
**`std::cin` is tied to `std::cout`**, so every
`std::getline(std::cin, …)` first forces a complete flush of `std::cout`.
Result: when the engine stops to wait for input, the entire text of the
question — including the final `> ` prompt — has already reached the GUI
through the pipe. The runner can therefore do:

```
accumulate stdout → buffer ends with "> "? → recognise the question → write the answer
```

### 1.4 The question→answer state machine

Every trajecta question contains a distinctive, stable phrase
("Enter maximum CPU threads", "Select cost function", …). The runner holds a
table of **rules**: `distinctive phrase → function producing the answer`
from the parameters collected in the form. When the buffer ends with `> `,
it looks up which phrase appears in the text accumulated since the last
answer and writes the corresponding answer to stdin.

**Error handling (loop guard).** If trajecta repeats the same question, its
internal validation rejected our input (e.g. "all points are outside the
DEM"). In that case the runner kills the process and shows the user the
`ERROR:` lines extracted from the transcript — a clean message instead of an
infinite loop.

**End of session.** After the computation trajecta asks "Run another
computation?" and then "Exit program?": the runner always answers `no` and
`yes`, so the process terminates cleanly with exactly one computation per
run.

### 1.5 Progress and results

- trajecta animates its progress bar by printing
  `\r\033[K\033[1m 45.3%\033[0m ███…`; a regular expression extracts the
  percentage and updates the GUI's `QProgressBar`.
- When the computation ends, the text between `Output Summary:` and
  `Run another` is extracted and shown in a summary card, together with the
  "Open output folder" button.

### 1.6 Automatic GDAL environment (usability for inexperienced users)

The engine needs the GDAL libraries (OSGeo4W). The installation guide used
to ask the user to manually add `OSGeo4W\bin` to the system PATH — the
hardest step for a non-technical user. The GUI removes it:

1. it looks for the GDAL DLLs next to `trajecta.exe` (advanced builds);
2. then in the folder chosen by the user ("Locate GDAL folder…");
3. then in every PATH entry;
4. finally in the standard locations `C:\OSGeo4W\bin`, `C:\OSGeo4W64\bin`, ….

The discovered folder is **prepended to the PATH of the child process only**
(no system changes), together with `PROJ_LIB` and `GDAL_DATA` when the
respective data folders (`share\proj\proj.db`, `apps\gdal\share\gdal`)
exist. The status is shown in the sidebar (✓/⚠) with explanatory tooltips.

### 1.7 Structure of the interface

Five pages in a `QStackedWidget`, navigated from the tabs in the top bar:

1. **Processing** — the whole of a single analysis, from the form to the
   console. Mode (FETE / LCPA / Batch, three selectable cards), input data
   (rows that change with the mode), cost modifiers (checkable group),
   algorithm, performance (defaults computed from the real hardware), output
   file names, the Run button, and directly below it the live run panel:
   status chip, progress bar, elapsed time, ANSI-aware console, summary card,
   Cancel / Pause / Open folder. Selecting the Batch card hides the single-run
   cards and shows the batch page in their place.
2. **Post-processing** — the NNI interpolation, with a run panel of its own.
3. **Viewer** — raster and vector display, colour scales, satellite basemap.
4. **Guide** — built-in user manual in HTML (modes, file requirements,
   parameters, GDAL, tips).
5. **About** — version, author, license, citations.

The run panel used to be a page of its own ("Processing", index 1). It was
folded into the setup page because the two are one task: `revealRunPanel()`
scrolls to it where `switchPage(1)` used to be called.

All fields are saved to `QSettings` on every run and reloaded at startup: a
user repeating similar analyses finds everything already filled in.

### 1.8 Preventive validation

Before launching the engine, the GUI checks everything it can check by
itself (files exist, extensions, output names without forbidden characters,
paths without non-ASCII characters the engine could not handle, output
folder created on request). What only GDAL can verify (georeferencing, CRS,
points inside the extent) is left to the engine — and its errors are
reported cleanly thanks to the loop guard.

### 1.9 Distribution (requirement 3)

The GUI source lives inside the Trajecta repository
(`gui/TrajectaStudio/`), so that `scripts/release.ps1`:

1. builds the GUI with Qt/MinGW;
2. runs `windeployqt` (copies every required Qt DLL into
   `build\gui_deploy`);
3. configures the engine project with `-DTRAJECTA_GUI_DIR=…`;
4. `cpack -G NSIS` produces **a single installer** `Trajecta-1.0.0.exe`
   containing engine + GUI + Qt runtime, with "Trajecta Studio" shortcuts in
   the Start Menu and on the Desktop and a "launch when finished" option.

The end user downloads a single .exe from GitHub, double-clicks it, and at
the end of the installation finds the interface ready to use.

---

## Part 2 — The code, file by file

> Convention: headings name the file; references like "around line N" point
> to the area of the file (exact numbers may drift slightly with each edit).

### 2.1 `main.cpp` — application bootstrap

```cpp
QApplication app(argc, argv);
```
Creates the Qt application object: event loop, DPI handling, arguments.

```cpp
QApplication::setOrganizationName("Trajecta");
QApplication::setApplicationName("TrajectaStudio");
```
These two lines determine where `QSettings` stores the preferences
(`HKEY_CURRENT_USER\Software\Trajecta\TrajectaStudio`).
`setApplicationVersion` uses the `TRAJECTA_STUDIO_VERSION` macro defined by
CMake (a single source of truth for the version). `setWindowIcon` loads the
icon from the compiled resource file (`:/assets/trajecta.ico`).

```cpp
QApplication::setStyle(QStyleFactory::create("Fusion"));
QApplication::setPalette(darkPalette());
```
`Fusion` is the style drawn entirely by Qt (identical on every Windows), so
it honours the dark palette. The local function `darkPalette()` sets every
color role (`Window`, `Base`, `Text`, `Highlight`, …) to the slate tones of
the logo: this is needed because some primitives (combo arrows, menus,
checkmarks) are drawn from the palette, not from the stylesheet.

```cpp
QFile qss(":/theme.qss");
app.setStyleSheet(...);
```
Loads the QSS theme from the resources and applies it globally.

```cpp
if (app.arguments().contains("--autorun"))
    QTimer::singleShot(0, &window, &MainWindow::triggerRun);
```
Hidden testing hook: with `--autorun` the GUI immediately starts the
analysis with the saved parameters. `singleShot(0, …)` defers the call until
the event loop has started (the window is already visible). A similar hidden
`--page setup|run|guide|about` switch opens a specific page.

### 2.2 `trajectarunner.h / .cpp` — the core: driving trajecta.exe

#### The `Parameters` struct
A plain struct with **all** the parameters existing in the engine's
interactive flow, using the same defaults as the engine: mode, verbose,
threads, RAM, the five input paths, the four cost-modifier fields, the three
algorithm parameters, the seven output names, plus the environment fields
(`exePath`, `gdalBinDir`, `projDataDir`, `gdalDataDir`, `workingDir`).

#### The regular expressions (top of the .cpp)

```cpp
const QRegularExpression kProgressRe("\\r\\x1B\\[K\\x1B\\[1m\\s*(\\d{1,3}(?:\\.\\d)?)%");
```
Recognises **exactly** the output of the engine's `print_progress()`
(`\r` + clear-line + bold + percentage). Being so specific, it cannot
mistake other percentages present in the log (e.g. "Avg CPU: 51.3%") for
progress.

```cpp
const QRegularExpression kAnsiRe("\\x1B\\[[0-9;?]*[A-Za-z]");
```
Strips every ANSI CSI sequence (colors, clear-line) to obtain the "clean"
text used for question pattern-matching.

```cpp
bool looksLikePrompt(const QString &s)
```
Returns `true` when the buffer, ignoring trailing spaces, ends with `>` at
the beginning of a line: the signature of the `> ` prompt the engine prints
before every `getline`. The "`>` preceded by a newline" check avoids false
positives on `>` in the middle of text.

#### `buildRules()` — the question→answer table

Each `add(key, lambda)` registers one rule. The keys are distinctive and
**mutually non-overlapping** phrases of the engine's 24 questions (both
modes). Non-obvious examples:

- `"for polyline rasterization"` vs `"for path smoothing"`: both questions
  start with "Select buffer radius (cells)", so the key is the ending part
  that distinguishes them.
- The numbers sent for the buffers **are the radius itself** (the engine
  code does `stoi(input)` and uses it directly), while for the neighbours
  the answer is the **menu index** (1→8, 2→16, 3→24, 4→32, 5→64): the lambda
  converts with a `switch`.
- `"Run another"` → always `no`, `"Exit program?"` → always `yes`: one GUI
  run = one computation.

The lambdas capture `this` and read `m_params`: no dangling pointers.

#### `start(params)` — launching the process

1. Resets the state (buffers, counters, flags) and rebuilds the rules.
2. Creates the `QProcess` with `SeparateChannels`: stdout carries the
   engine's prompts, charts and progress bars, while stderr (unbuffered
   GDAL/PROJ error lines) is read separately, split into complete lines and
   emitted as `consoleErrorLine` — so an error can never splice itself into
   the middle of a Unicode chart. Both channels are decoded with stateful
   `QStringDecoder`s, so a multi-byte UTF-8 character split across two pipe
   chunks no longer becomes a � replacement character.
3. **Builds the child environment**: starts from the system one and prepends
   `gdalBinDir` to `PATH`; exports `PROJ_LIB` and `GDAL_DATA` if detected
   **and only if the user has not already set them** (never override the
   user's explicit choices).
4. `setWorkingDirectory(workingDir)`: trajecta saves `fete_config.txt` /
   `lcpa_config.txt` in the current directory; the GUI points it to a
   writable AppData folder, keeping both Program Files and the results
   folder clean.
5. Connects the three process signals (`readyReadStandardOutput`,
   `finished`, `errorOccurred`) and starts `trajecta.exe` with no arguments.

#### `onReadyRead()` — for every output chunk

1. Decodes the bytes as **UTF-8** (the engine prints Unicode bars).
2. `emit consoleOutput(raw)`: the raw text goes to the GUI console.
3. Applies `kProgressRe` and emits `progressChanged` with the last
   percentage in the chunk.
4. Strips ANSI, normalises `\r`→`\n`, accumulates in two buffers:
   `m_pending` (since the last answer) and `m_fullLog` (the whole
   transcript).
5. Looks for the last line of the chunk starting with an activity verb
   (`Loading|Computing|Rasterizing|…`) and emits it as `statusChanged` (the
   phase line under the status chip).
6. Calls `handlePendingOutput()` → if `looksLikePrompt(m_pending)` then
   `answerPrompt()`.

#### `answerPrompt()` — the heart of the state machine

```cpp
for (const PromptRule &rule : m_rules) {
    const int pos = m_pending.lastIndexOf(rule.key);
    if (pos > bestPos) { bestPos = pos; bestRule = &rule; }
}
```
Among all rules whose key appears in the buffer, the one **closest to the
prompt** (highest position) wins: this protects against the first buffer,
which contains the banner with potentially ambiguous words.

- No rule found → `abortRun("unexpected question…")`: if a future engine
  version adds questions, the user gets a clear error instead of a hang.
- `++m_askCount[key] > 1` → **loop guard**: the question was repeated, so
  the engine rejected the input; the error lines are extracted with
  `errorLinesFromLog()` and the run is aborted.
- Otherwise: `m_process->write(answer.toLocal8Bit() + '\n')` — the
  local-8-bit encoding replicates exactly what the engine would receive from
  a Windows console — and `emit answerSent(answer)` echoes it to the GUI
  console.

#### `errorLinesFromLog()` / `extractResultReport()`

The first collects the lines starting with `ERROR`/`WARNING` or containing
"Please correct" (the last 6: the most recent ones describe the actual
failure). The second cuts the text between `Output Summary:` and
`Run another`: the engine's official summary (timings, statistics, file
list).

#### `onProcessFinished(exitCode, status)`

Order of the cases: cancelled by the user → aborted by the loop guard →
`successfully computed!` marker present → success (progress to 100, report
extracted). Otherwise it tries to explain the failure; the special case
`0xC0000135` (STATUS_DLL_NOT_FOUND) produces the "GDAL not found, install
OSGeo4W or use Locate GDAL folder" message.

#### `onProcessError`

Only `FailedToStart` is handled here (the exe does not exist / is not
executable): real crashes still go through `onProcessFinished`.

#### `pause()` / `resume()` — freezing the engine

`pause()` suspends every thread of the child process in one shot through
`NtSuspendProcess` (ntdll; undocumented but stable for decades — Process
Explorer uses it), `resume()` thaws it with `NtResumeProcess`. The CPU is
released immediately while the engine's RAM stays allocated: the paused run
survives system sleep and hibernation, but not a shutdown. `cancel()` and
`abortRun()` always resume before killing so termination is delivered
promptly, and `onProcessFinished` clears a stale paused flag if the process
dies while frozen. The GUI listens to `pauseStateChanged` to toggle the
Pause/Resume button, show the PAUSED chip and stop the elapsed clock (time
spent paused is subtracted, so the label shows working time only).

### 2.3 `consoleview.h / .cpp` — live console with ANSI

Extends a read-only `QPlainTextEdit`, monospace font (Cascadia Mono with
Consolas fallback), `setMaximumBlockCount(8000)`: the document self-prunes,
memory stays bounded even on very long runs.

`appendChunk(raw)` is a small character-level parser:

- `ESC [ … m` → `applySgrCodes()`: maps the SGR codes (0 reset, 1 bold,
  31/32/33/36/90/… the colors the engine uses) onto `QTextCharFormat`, so
  cyan questions, green successes and red errors appear as in the real
  console.
- `\r` → raises the `m_pendingCr` flag: the **next** printed text will
  replace the current line (`insertRun` selects from the start of the block
  and removes). This is exactly how a console animates the progress bar,
  without adding thousands of lines to the log.
- `\n` → closes the current block (`insertBlock`).
- The other CSI sequences (e.g. `K` clear-line) are ignored: their effect is
  already covered by the `\r` handling.

Autoscroll is smart: it follows the tail only if the user was already at the
bottom (`atBottom()`), so the log can be scrolled during the computation
without being yanked back down.

`appendMarker(text, color)` inserts GUI-generated lines (answers sent,
"Launching…", outcomes) in colored bold, distinguishable from the engine
output.

### 2.4 `pathpicker.h / .cpp` — path selector

Reusable widget composed of: **status dot** + `QLineEdit` (with clear
button) + *Browse…* button. Two kinds (`ExistingFile` / `Directory`),
configurable dialog filter, `optional` flag.

- `updateIndicator()`: gray dot when empty, green when the path exists and
  has the right kind, red otherwise — immediate feedback without pressing
  Run.
- `isSatisfied()`: used by the MainWindow validation (an empty optional
  field is "satisfied").
- `dragEnterEvent`/`dropEvent`: accepts files dragged from File Explorer
  (`mimeData()->urls()` → `toLocalFile()`).

### 2.5 `systeminfo.h` — total RAM

`GlobalMemoryStatusEx` (WinAPI) for the physical RAM in MB, with an 8 GB
fallback on non-Windows platforms. `NOMINMAX` before `windows.h` prevents
the `min`/`max` macros from breaking the standard headers (same precaution
as the engine's CMake). Also holds `kRecommendedRamMb` (4096), the memory
ceiling offered on every machine — the detected total is only used to show
what is installed and to cap the recommendation on a small system.

### 2.6 `mainwindow.h / .cpp` — the interface

#### Constructor
Root layout = `QHBoxLayout` without margins: fixed sidebar (220 px) +
`QStackedWidget` with the 4 pages. Then the **runner wiring**: five
`connect`s hook the signals (`consoleOutput→appendChunk`,
`answerSent→appendMarker`, `progressChanged→setValue(pct*10)` — the bar has
a 0-1000 range to show decimals —, `statusChanged→phaseLabel`,
`finished→onRunFinished`). A 1-second `QTimer` updates the elapsed time.
Finally `loadSettings()`, `updateModeUi()`, `updateEnvironmentStatus()`.
On Windows the title bar is switched to dark via
`DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE)`.

#### `buildSidebar()`
Logo, title, version, four checkable `NavButton`s in an exclusive
`QButtonGroup` (the `idClicked(int)` signal switches pages), then at the
bottom the ENVIRONMENT section with the two indicators (engine/GDAL) and the
link-style "Locate engine…" / "Locate GDAL folder…" buttons.

#### `makeCard(title, subtitle, content)`
The card factory: a `QFrame` with `objectName("Card")` (the QSS gives it
background, border and rounded corners) containing a title, a gray subtitle
and the content widget. Every form section is a card → consistent look.

#### `buildSetupPage()` — the complete form
A `QScrollArea` containing, in order:

1. **Analysis mode**: two multi-line checkable `QPushButton`s (`ModeCard`)
   in an exclusive group; clicking calls `updateModeUi()`. Their size policy
   is `Ignored` horizontally so they share the row equally.
2. **Input data**: `QGridLayout` with five `PathPicker`s (DEM, Sample
   points, Origin, Destinations, Output folder). The local `addRow` lambda
   creates label+widget and returns the label: `updateModeUi()` uses it to
   show/hide the right rows (points for FETE; origin+destinations for LCPA).
3. **Cost modifiers**: a **checkable** `QGroupBox` — when unchecked Qt
   automatically disables all children. Inside: vector picker (optional),
   polyline buffer spin (0–10, default 2), raster picker (optional), and the
   barrier row composed of a "Multipliers above" checkbox + `QDoubleSpinBox`
   (default 1000) + "are impassable barriers" text; the checkbox
   enables/disables the spin. Every row has a tooltip explaining the
   parameter with the same wording as the engine's guide.
4. **Algorithm**: neighbours combo (the 5 values with description; the real
   value 8/16/24/32/64 is in the `userData`), cost function combo (3
   entries, `userData` 1–3), smoothing buffer spin.
5. **Performance**: threads = `idealThreadCount()-4` (min 1), from the
   hardware as the engine does. RAM is *not* derived from the machine: the
   ceiling is `SystemInfo::kRecommendedRamMb` (4096 MB), capped by what is
   installed, because the working set is the same on every machine and a
   larger ceiling buys nothing. A stored value equal to the old 60%-of-
   physical default is migrated to the new one on load. The hint labels show
   the detected values. Verbose checkbox.
6. **Output files**: seven `QLineEdit`s laid out in two columns with the
   manual's defaults (`slope`, `cost_surface`, …). The `addName` lambda
   returns the (label, edit) pair via `std::tie`, so `updateModeUi()` can
   hide density (LCPA) or paths raster/shape (FETE).
7. **Run bar**: the big orange `RunButton`.

At the end, `guardWheel()` is applied to every combo/spin: an event filter
that ignores wheel events when the widget is not focused, so scrolling the
form can never silently change values.

#### `updateModeUi()`
Eight `setVisible()` calls reconfigure the form according to the mode. The
`QGridLayout` automatically compacts the hidden rows.

#### `buildRunPage()`
Status row (chip + phase + time), 0–1000 `QProgressBar`, expanded
`ConsoleView`, hidden summary card (`m_summaryCard`), button row (Back /
Open output folder / Pause-Resume / Cancel run with confirmation). The chip
uses a **dynamic property** `state` (idle/running/paused/success/failed):
the QSS has one selector per value, and after `setProperty` a simple
`unpolish/polish` refreshes the style.

#### `buildPostPage()` — NNI post-processing
Form for the engine's mode 3 (Natural Neighbour Interpolation, discrete
Sibson): density raster picker (prefilled automatically after a successful
FETE run), output folder, sample threshold, max search radius (0 =
unlimited, shown as special value text) and output name. "Run
interpolation" validates, overrides `Parameters` with `Mode::Interp` + the
`interp*` fields and funnels into the same `beginRun()` path as a normal
analysis, so progress, console, pause and cancel work identically.

#### `buildGuidePage()` / `buildAboutPage()`
`QTextBrowser` with static HTML (external links openable) for the guide;
About page with logo, version, author, license and citation.

#### Environment discovery
- `engineExePath()`: priority to (1) the override saved by "Locate
  engine…", (2) the `TRAJECTA_ENGINE` environment variable (handy in
  development), (3) `trajecta.exe` next to the GUI (the installer layout).
- `detectGdalEnvironment()`: implements the cascade described in §1.6 and
  collects the candidate "roots"; from these it also resolves
  `share/proj/proj.db` (→ `PROJ_LIB`) and `apps/gdal/share/gdal` or
  `share/gdal` (→ `GDAL_DATA`). Returns a
  `{found, binDir, projData, gdalData}` struct.
- `updateEnvironmentStatus()`: translates the struct into the two ✓/⚠
  sidebar indicators, with the path in the tooltip.

#### `validationError()`
Returns the **first** thing to fix (empty string = all good): required files
existing according to the mode, modifiers existing when the group is
enabled, output names non-empty and free of `\/:*?"<>|`, ASCII-only paths
(the engine receives bytes in the local codepage: accented characters would
cause mysterious failures — better a clear message before starting).

#### `collectParameters()`
Transfers the widgets into the `Parameters` struct. Note the normalisation:
if the modifiers group is enabled but both paths are empty,
`useCostModifiers` becomes `false` (the engine's question will be answered
"no", skipping the sub-questions). A disabled barrier sends `0`, which the
engine interprets as "threshold disabled". `workingDir` points to AppData.

#### `startRun()`
Sequence: validation (warning dialog on failure) → offer to create the
output folder if missing → engine check (dialog with instructions if
missing) → `saveSettings()` → run panel reset (clean console, "Launching…"
marker, bar at 0, RUNNING chip, cancel enabled, run disabled, stopwatch
started) → `revealRunPanel()` (scrolls the setup page down to the panel) →
`m_runner->start()`.

#### `onRunFinished(success, report)`
Stops the stopwatch, re-enables Run, disables Cancel, sets chip/phase/card
title according to the outcome (success / cancelled / failed), writes the
report into the card inside an HTML-escaped `<pre>` (preserving the
alignment of the engine's summary), enables "Open output folder" only on
success, and appends the final colored marker to the console.

#### Persistence
`saveSettings()`/`loadSettings()` read/write every form field plus the
window geometry in `QSettings` (user registry). Empty output names in the
registry never override the defaults (`loadName` checks).

#### `closeEvent()`
If an analysis is running it asks for confirmation, then cancels the process
and saves.

### 2.7 `theme.qss` — the theme

Palette derived from the logo: slate `#12161d`/`#1a2029` (backgrounds),
green `#3fa34d` (accent: selections, focus, checkboxes, ok state), orange
`#ef8b2e` (primary action: Run, progress bar). Main sections:

- `QFrame#Sidebar` / `NavButton`: dark navigation with the checked state
  highlighted by a background + green left border.
- `QFrame#Card`: the form cards (border `#262f3b`, 10 px radius);
  `QFrame#Card QWidget { background: transparent }` prevents children from
  repainting the background.
- Inputs (`QLineEdit`, spins, combos): dark background, border turning green
  on focus; combo popup styled via `QComboBox QAbstractItemView`.
- `QPushButton#RunButton` orange with dark text (AA contrast),
  hover/pressed/disabled states; `#DangerButton` muted red for Cancel.
- `QLabel#StateChip[state="…"]`: four colored chip variants driven by the
  dynamic property.
- `QProgressBar::chunk` with an orange gradient; console `#0b0f14`; minimal
  scrollbars without arrows.

### 2.8 `resources.qrc`, `app.rc`, assets

`resources.qrc` embeds the theme, logo and icon **inside the executable**
(no external files to distribute). `app.rc`
(`IDI_ICON1 ICON "assets/trajecta.ico"`) is the Windows resource giving the
.exe its icon in File Explorer.

### 2.9 `runner_probe.cpp` — testing without the GUI

`QCoreApplication` + `QCommandLineParser`: builds a `Parameters` from the
CLI arguments, hooks the runner signals to simple prints and exits with
code 0/1 according to the outcome. It is used to test the state machine
against the real `trajecta.exe` in a scriptable way (this is how the
interface was tested end-to-end, including the "points outside the DEM"
error case that verified the loop guard). Built only with
`-DTRAJECTA_STUDIO_BUILD_TESTTOOL=ON`, never distributed.

### 2.10 `CMakeLists.txt` (GUI)

`qt_add_executable(TrajectaStudio WIN32 …)`: `WIN32` = windowed application
without a console. `CMAKE_AUTOMOC/AUTORCC` automatically generate the
meta-object code from `Q_OBJECT` and compile the `.qrc`. The project version
becomes the `TRAJECTA_STUDIO_VERSION` macro. Optional `runner_probe` target
(Core only). `install(TARGETS … DESTINATION .)` for CPack. On Windows the
target also links `dwmapi` (dark title bar).

### 2.11 Installer integration

**Engine `CMakeLists.txt`** — new cache variable `TRAJECTA_GUI_DIR`: when
set, it verifies that the folder contains `TrajectaStudio.exe` and installs
the whole deployed folder into the package root. CPack version bumped to
1.0.0. With the GUI present: `CPACK_PACKAGE_EXECUTABLES` (Start Menu
shortcut), `CPACK_CREATE_DESKTOP_LINKS` (desktop),
`CPACK_NSIS_EXECUTABLES_DIRECTORY "."` (the executables live in the install
root, not in the default `bin\`), `CPACK_NSIS_MUI_FINISHPAGE_RUN` (launch
when the installer finishes).

**`scripts/release.ps1`** — new parameters `-QtDir`, `-MinGWDir`,
`-NinjaExe`, `-SkipGui`. Phase 1: configures and builds the GUI (Ninja if
available, otherwise MinGW Makefiles), copies the exe into
`build\gui_deploy` and runs `windeployqt --release --compiler-runtime`
(copies the Qt DLLs, platform plugins and MinGW runtime). Phase 2: as
before, but passes `-DTRAJECTA_GUI_DIR` to the engine configure;
`cpack -G NSIS` packages everything and the installer's SHA256 is computed.

---

## Part 3 — Map: engine questions → GUI widgets

| trajecta.exe question | Widget | Answer sent |
|---|---|---|
| Choose computation mode | FETE / LCPA cards | `1` / `2` |
| Enable detailed debug output? | *Detailed debug output* checkbox | `yes` / `no` |
| Enter maximum CPU threads | *CPU threads* spin | number |
| Enter maximum RAM to allocate (MB) | *Maximum RAM* spin | number |
| Enter path to DEM file | *DEM raster* picker | path |
| Enter path to sample points file | *Sample points* picker (FETE) | path |
| Enter path to ORIGIN file | *Origin* picker (LCPA) | path |
| Enter path to DESTINATIONS file | *Destinations* picker (LCPA) | path |
| Enter output directory | *Output folder* picker | path |
| Do you want to add additional cost modifiers? | *Cost modifiers* group | `yes` / `no` |
| Enter path to cost modifiers vector file | *Vector modifiers* picker | path or blank |
| Select buffer radius … polyline rasterization | *Polyline buffer* spin | radius |
| Enter path to cost modifiers raster | *Raster modifiers* picker | path or blank |
| Treat extreme cost multipliers as impassable barriers | *Barrier threshold* checkbox+spin | threshold or `0` |
| Select number of neighbours | *Neighbours* combo | index `1`–`5` |
| Select cost function | *Cost function* combo | `1`–`3` |
| Select buffer radius … path smoothing | *Path smoothing buffer* spin | radius |
| Enter slope raster filename | *Slope raster* field | name |
| Enter base cost surface raster filename | *Base cost surface* field | name |
| Enter additional cost surface raster filename | *Additional cost* field | name |
| Enter total cost surface raster filename | *Total cost surface* field | name |
| Enter output density raster filename | *Density raster* field (FETE) | name |
| Enter path raster filename | *Paths raster* field (LCPA) | name |
| Enter path lines shapefile filename | *Paths shapefile* field (LCPA) | name |
| Run another …? / Exit program? | — automatic | `no` / `yes` |

---

## Part 4 — The interface as it is now (1.0.1)

Parts 1–3 describe the interface as it was first built: a sidebar, one page per
job, one form. The application has since grown a top bar, a batch mode, a map
viewer, checkpointing and a guided tour, and this part covers the code behind
those. Where it contradicts Part 2, this part is the current one.

### 4.1 The shape of the window

`MainWindow` is a frameless window on Windows (`Qt::FramelessWindowHint` plus a
`nativeEvent` that answers `WM_NCHITTEST`, so edge resizing, Aero Snap and
maximise-respects-the-taskbar all still work) and an ordinary native window
everywhere else. Inside it there are three things: `buildTopBar()`,
a `QStackedWidget` of five pages, and `buildStatusBar()`.

The five pages are built once, in the constructor, and never rebuilt:

| Page | Built by | What lives there |
|---|---|---|
| Processing | `buildSetupPage()` | mode cards, the whole single-run form, `BatchPage`, and the run panel |
| Post-processing | `buildPostPage()` | the NNI card, the comparison card, their panels |
| Viewer | `ViewerPage` | the map canvas and its controls |
| Guide | `buildGuidePage()` | a `QTextBrowser` with the manual |
| About | `buildAboutPage()` | version, licence, contacts |

Two consequences worth knowing. First, every widget the walkthrough points at
exists from start-up, whether or not its page has ever been shown — which is
what lets the tour be built as a flat list of pointers. Second, "switching mode"
is not a page change but a visibility change: `updateModeUi()` shows or hides
the cards collected in `m_singleRunCards` and, for batch, swaps the entire
single-run form for `BatchPage`.

### 4.2 `uiwidgets.h / .cpp` — the pieces used everywhere

Small shared builders, so that a control that appears in two places is the same
control and not a copy that drifts:

- `makeHelpDot(text)` / `withHelpDot(widget, text)` — the `?` badge and its
  pop-up. Every parameter in the application explains itself through one of
  these, which is why the form has almost no inline prose.
- `makeGroupNote(group, note, help)` — a caption beside a checkable group's
  title, on the title's own line and starting just after it. Where the title
  ends is asked of the style (`subControlRect(SC_GroupBoxLabel)`) and asked
  again on every font and theme change, because the title is drawn by the style
  from a stylesheet that indents it, in a font that is a user setting, in text
  that is translated — no constant could be right. A checkable `QGroupBox` also
  greys out its contents while unticked, which is exactly when a warning about
  what ticking it costs has to be readable, so the row puts itself back on every
  toggle.
- `setPauseMark(button, on)` — the two bars on a *Pause* button. Painted, not
  typed: the ▶ it answers to is the glyph U+25B6 inside the button's label, and
  no interface font Trajecta offers contains it, so its size and weight come
  from whatever fallback face the platform picks. The bars are therefore built
  from the ink that glyph actually puts on screen at that button's font
  (`tightBoundingRect`), with square corners to match its sharp ones, and are
  repainted on every theme and font change by a keeper object living on the
  button. `on == false` takes the mark off, which is what a Pause button needs
  when it becomes "▶ Resume": the mark leaves with the word.
- `ActivityBar` — the progress bar. It overrides `text()` to show one decimal
  only when the format is `%p%`, because "37%" for four hours of work looks
  stuck while "37.4%" visibly moves.
- `kLogCanvasHeight` — one constant for every log canvas in the application, so
  the batch transcript and the single-run one are read the same way.

### 4.3 `batchpage.h / .cpp` — the queue

A batch is a list of **chunks**; a chunk is a group of **rows** that share an
algorithm and a set of cost modifiers. `BatchChunkWidget` is one chunk: the
algorithm controls at the top, the modifiers group, and a `BatchTableView` over
a `BatchTableModel` for the rows. `BatchPage` owns the chunks, the hardware
settings that apply to all of them, and the run panel.

Three points that are easy to get wrong and were:

- **The page stays editable while it runs.** Only *Run batch* is disabled
  (`setStartAllowed(false)`); rows further down the queue can still be added or
  corrected while the ones above them compute. The editing lock exists
  separately (`setEditingEnabled`) and is used only where it is genuinely
  needed.
- **Every chunk carries at least one row.** The rule lives in `addChunk()`,
  the single door a chunk comes through — added, duplicated, restored at
  start-up, or loaded from a file.
- **The batch is saved as you type.** `saveState()` / `restoreState()` put the
  whole queue into `QSettings` as JSON, so a hand-typed table survives closing
  the window without an explicit *Save batch*.

`BatchController` drives the queue: it holds a `TrajectaRunner`, feeds it one
row at a time, and reports progress across the whole job rather than the row.

### 4.4 `viewerpage.h / .cpp` — the map

The Viewer is the only part of the interface that talks to GDAL directly, which
it does through `GdalApi` — a hand-bound subset of the GDAL **C** API loaded at
runtime with `LoadLibrary`/`dlopen`. The C ABI is the one interface that is safe
to share between the MinGW-built GUI and the MSVC-built OSGeo4W libraries, which
is why not one C++ GDAL symbol is used anywhere.

**Rasters.** `registerRaster()` adds a layer without reading it; the file is
opened on first selection. `loadLayer()` reads a decimated display buffer
(capped at 4096 px on the long side), the statistics, a sorted sample for
percentiles, and a log-scaled histogram for the filter slider. The full-size
dataset stays open only for the exact value under the cursor.

**Vectors.** `registerVectorOverlay()` adds an overlay; `loadOverlayGeometry()`
reads it. Points, lines, polygons (as their rings), multi-geometries and
geometry collections are all collected, recursively and depth-limited, because
the structure comes from a file and not from us. A CSV gets open options that
make GDAL's driver produce geometry from coordinate columns — without them a
table of coordinates opens as attributes and draws nothing, which is the
commonest way a text import silently "works" and shows an empty map.

**Where the geometry is drawn.** Overlays are kept in the CRS they were read in
and pushed into the CRS of the raster underneath at draw time
(`projectOverlay()`, cached per target CRS). Two details there are load-bearing:

- `OCTTransform`'s return value is *ignored*. It answers "did every point
  convert", and says no as soon as one falls outside the area the
  transformation is defined for — while still converting all the others and
  marking only the failures with `HUGE_VAL`. Treating that as a failed call
  throws away a whole layer because one point sits off the edge of the
  projection. The check that matters is per point: `std::isfinite`.
- Bounding boxes are accumulated as four numbers (`BoundsAccumulator`), never
  with `QRectF::united()`. A rectangle around a single point has no width and
  no height, which is what Qt calls *null*, and `united()` answers a null
  rectangle by ignoring it — so a layer of points would never grow a box at
  all. This cost an hour before it was found, and it is the reason the class
  exists.

**A vector on its own** gets a scene of its own: with no raster to hang it on,
`rebuildOverlay()` frames the shared extent of everything ticked into a
~1600-unit scene and sets the scalebar from that. Without it, importing a
vector into an empty Viewer would land on the "no layer loaded" placeholder and
look like a failed import.

**Opening a file** goes through `openAnyFile()`, which asks GDAL to open it as a
raster and then as a vector, and lets the content decide. An extension is a hint
about a file, not a fact.

### 4.5 `checkpointstore.h / .cpp` — surviving an interruption

The engine writes the propagation state (§5.6); that is enough to carry on
computing but not enough to *start* the engine again — the DEM, the output
names, the algorithm and, in a batch, the rows still to run all live in the
interface. So a run with checkpointing on drops a `session.json` next to the
checkpoint and removes it when the run ends in any orderly way. A session file
still present at the next start therefore means one thing only: the last run did
not get to say goodbye. That is the whole basis of the recovery prompt.

`copyTo()` / `exportTo()` / `importFrom()` move that pair of files about.
`exportTo()` removes exactly what `copyTo()` reported having copied, never a
fresh listing of the folder — re-reading it could delete a checkpoint written
after the copy was taken, and that is a file nobody has a second copy of.

The two links on every log row (*Save a copy of the checkpoint…*, *Resume from a
checkpoint file…*) are the deliberate half of this: before them, an interrupted
analysis could only be recovered by accident of a crash, on the same machine,
once.

### 4.6 `walkthrough.h / .cpp` — the guided tour

The engine of the tour knows nothing about Trajecta. It is handed a list of
`TourStep`s — widgets to light, a title, a body, optional captions with leader
lines, an optional still picture, and an optional `onEnter` callback — and it
iterates them. Everything Trajecta-specific lives in the code that builds the
list: `MainWindow::buildWalkthrough()`, `BatchPage::walkthroughSteps()` and
`ViewerPage::walkthroughSteps()`, the last two because the widgets they point at
are private to their pages.

`TourOverlay` is a child widget covering the whole window. It:

- paints a scrim over everything and cuts a rounded hole around the union of the
  step's target rectangles, with concentric strokes of falling alpha for the
  glow;
- **blocks everything**. Every mouse, wheel, context-menu and key event is
  accepted and dropped; it accepts drops and ignores them; and a `qApp` event
  filter swallows `KeyPress`, `KeyRelease` and `ShortcutOverride` so not even a
  shortcut reaches the application underneath — the two exceptions being ← and
  →, which the filter uses itself to turn the pages;
- measures a target by `mapFromGlobal(mapToGlobal(...))` — not `mapTo()`, which
  requires the target to be a descendant — and intersects the result with every
  scroll viewport above it, so a widget scrolled half out of view is lit only
  where it is actually visible;
- settles the layout before measuring: `onEnter` first, then
  `layout()->activate()` and `sendPostedEvents(LayoutRequest)`, then, on the
  next turn of the event loop, `ensureTargetVisible()` and the measurement.

**What a step lights is a whole card**, not the controls inside it:
`TourStep::lightCard()` sets the target, drops the padding to nothing and takes
the corner radius from `ThemeManager::cardRadius()`, so the cut-out is the card
— its heading, its note and its fields — in the card's own shape. Half of what a
screen says is *which part of the page* this is, and the heading is the half
that says it. Controls that are not part of a section (the Run button, Export)
are still lit on their own.

Two consequences followed from that:

- **The scrolling had to be done by hand.** `QScrollArea::ensureWidgetVisible()`
  scrolls the least it can, which leaves a card taller than the viewport resting
  its top edge on the bottom of the window. `ensureTargetVisible()` centres what
  fits and top-aligns what does not — and reports whether it moved anything, so
  the step is measured on the *next* turn of the event loop when it did. Widget
  positions after a scroll are not reliably settled inside the same turn, and
  measuring through that window lit the card above the intended one, with every
  caption pointing into the gap. It re-measures at most three times: a chunk
  unfolding on the batch page keeps moving for 180 ms, and the count is what
  stops "it moved again" from becoming a loop.
- **The navigation bar moves out of the way.** A card that reaches the bottom of
  the window had the ‹ 24 / 44 › bar drawn across its last row, so
  `placeNavBar()` is given the lit rectangle and goes to the top of the window
  instead when the two would meet. Everything else that has to keep off it —
  callout, captions, inset — asks `topLimit()` / `bottomLimit()` rather than
  assuming the bottom.

Three problems were solved in ways worth recording:

- **The gear menu cannot be opened.** A `QMenu` is a top-level popup that takes
  a mouse and keyboard grab: opened during the tour it would paint over the
  overlay, stay clickable, and swallow the click meant for *Continue*. So
  `renderMenuPicture()` renders it offscreen (`WA_DontShowOnScreen` + `grab()`)
  and the step draws the picture where the real menu would drop.
- **Captions.** Their arrangement follows the shape of what they point at, not
  the free space: controls spread across the page get a row of captions
  underneath, controls stacked in a column get a column beside. A forward cursor
  places them left to right, and a second pass right to left pulls the tail back
  inside the window — without it a row of five under controls near the right
  edge hangs off the screen while a gap opens in the middle.
- **Transitions.** A step change is a cross-fade, not a cut: the old panel fades
  out (110 ms), the page changes and the layout settles while nothing is being
  read, and the new panel fades in (170 ms) with a 10 px rise. The spotlight
  glides between rectangles on an `InOutCubic` curve over 300 ms. The callout
  width is chosen per step (`fitCalloutWidth()`) as the narrowest of a set of
  candidates at which the whole text is *comfortable* — a height bound stricter
  than "it fits", because a paragraph poured into a narrow column becomes a tall
  ribbon of six-word lines long before it runs off the screen.
- **Nothing of the previous step survives it.** Every repaint the overlay asks
  for while the light is travelling is a partial one, computed from where the
  light was and where it is going. The captions, their leader lines and the
  inset picture lie outside that region, so on a step change nothing ever
  invalidated them and they stayed on screen over the new screen — until some
  unrelated event forced a full repaint, which is why clicking outside the
  window appeared to "fix" it. `measureAndShow()` now asks for one whole repaint
  per step change; the partial ones keep the job they exist for, which is the
  sixty a second while the spotlight moves.
- **Every screen stands on its own.** A step that relied on the page the
  previous one left behind lights nothing when it is reached backwards — and the
  tour is walked backwards as often as forwards. Each block of screens therefore
  ends with `closeBlock()`, which hands its own navigation to every screen in
  the block that did not set one.
- **The panel is the same shape on every screen.** Its heading wraps, and a
  wrapping `QLabel` is handed the height it *guesses* it will need, not the
  height its text takes at the width it gets; the surplus is then split above
  and below by the vertical centring. `fitCalloutWidth()` therefore pins the
  heading with `heightForWidth()` at a width it works out itself
  (`w − 2·padH − 2·closeWidth − 2·headSpacing`) rather than reading
  `m_title->width()`, which is only true after the layout has run at the new
  width — reading it measured the *previous* screen's width, which is why some
  panels gaped and others were cramped. Two more things had to go with it: the
  spacer that balances the ✕ is `Fixed` vertically instead of the growable one
  `addSpacing()` makes, or every spare pixel in the panel ends up in the heading
  row; and the panel's height is `totalHeightForWidth(w)` rather than
  `adjustSize()`'s hint, which over-estimates a paragraph by a line or two at
  the wider settings. What is left is one constant, `kTitleGap`, used for the
  air above the heading, below it, and under the paragraph.
- **← and → turn the pages.** Caught in the `qApp` filter, the only place they
  pass through, since the overlay never takes the focus. Press only, no
  auto-repeat, and the keypad modifier is masked off before the "no modifiers"
  test: Windows reports the arrow keys as extended keys and Qt turns that into
  `KeypadModifier`, so an arrow arrives "modified" with nothing held down.
- **Back and Continue are one pair.** `matchFootButtons()` gives both whichever
  of the two needs more room, in both directions, at every step — their labels
  differ, "Continue" becomes "Finish" on the last screen, and the filled one is
  styled in a larger, heavier face than the outlined one, so their size hints
  never agreed.
- **The ‹ › and the counter sit on the bar's axis.** Text is centred by its line
  box — ascent above the baseline, descent below — and a chevron or a row of
  digits has nothing under the baseline, so the ink rides low in it: almost four
  pixels for the marks, one and a half for the counter. `inkDrop()` measures the
  real ink with `tightBoundingRect()` and `centringCss()` takes twice that off
  the bottom padding, which raises centred text by exactly it. Measured rather
  than nudged by a constant because the interface font is a setting, and
  truncated rather than rounded because the rasterised ink sits a fraction above
  the metric one.
- **The count is inside the panel; the navigation bar is switched off.** It sits
  between Back and Continue, which are one pair of the same width, so two equal
  stretches put it exactly at the panel's centre. The bar it came from —
  `‹ 7 / 44 ›`, floating over the page — is kept whole behind `kShowNavBar`,
  including its own copy of the counter, so it can be brought back by flipping
  one line; its two arrows only ever repeated what the two buttons and the
  ← → keys already do. With the bar off, `topLimit()`/`bottomLimit()` stop
  reserving 76 px for it and every screen gets the extra room.
- **An unavoidable overlap still shows the lit edge.** A card that fills the
  window leaves the panel nowhere to stand, and drawn flush against the card's
  top the panel covered the frame — at which point the light stopped reading as
  an outline around a section. `kFrameReveal` keeps 20 px of the lit edge in
  sight on whichever side the panel is pushed to.

The rule the whole thing is built around: **the tour changes nothing.**
`startWalkthrough()` photographs the page, the mode and the post-processing tool
before it moves anything, and `restoreAfterWalkthrough()` puts all three back —
including taking the two example layers out of the Viewer. `closeEvent()` calls
that restore *before* `saveSettings()`, because otherwise closing the window on
the batch screen would leave Batch as the mode that comes back next time.

Leaving is a question, not a click. The ✕ emits `closeRequested()` rather than
closing: the overlay does not know what to ask or what to say afterwards, so
`MainWindow::confirmCloseWalkthrough()` asks, and on yes says where the tour can
be found again — the Guide page, which is exactly the thing a user who has just
left it has not been told. That means a modal dialog in front of an overlay
whose `qApp` filter eats every key, so the filter now stands aside while
`QApplication::activeModalWidget()` is up: a dialog whose buttons answer to
neither Enter nor Escape is broken.

### 4.7 The example layers

The Viewer block of the tour needs something on the canvas. The two files live
inside the executable as Qt resources (`assets/tour/`), are unpacked into a
`QTemporaryDir` when the block is entered, and are taken away at the end. The
order on the way out is not negotiable: unload the layers, which closes their
GDAL datasets, and only then delete the files — on Windows an open file cannot
be removed and the failure is silent. `~ViewerPage` does the same by hand,
because `m_tourDir` is declared last and would otherwise be destroyed *before*
the datasets it holds.

---

## Part 5 — The propagation algorithm, and how Trajecta implements it

This part is about the engine (`src/`), not the interface. It is here because
every setting in the form is an argument to what follows, and the form cannot be
understood without it.

### 5.1 The model

A DEM is turned into a **weighted directed graph**. One node per cell; one edge
for each move the neighbourhood template allows; the weight of an edge is the
cost of making that move. Everything Trajecta computes is then a shortest-path
problem on that graph.

The graph is **directed** and this is not a formality: walking up a slope and
walking down it cost different amounts, so the edge *v→u* and the edge *u→v*
carry different weights. It is why the corridor computation (§5.7) has to run a
second, reversed search rather than reuse the first.

### 5.2 The neighbourhood (`src/neighbourhood.h`)

The set of moves allowed from a cell cannot be chosen freely. It has to be
closed under the eight symmetries of the square — four rotations and their
mirrors — or some compass directions become cheaper to travel than others and
every result leans that way. So offsets come in whole orbits under that group:
eight members in general, four when the direction lies on an axis (a, 0) or a
diagonal (a, a), because those map onto themselves.

That is why the neighbourhood sizes offered are 8, 16, 24, 32, 40, 44, 48… and
not "37": the reachable sizes are the running totals of those orbits. Within a
Chebyshev ring the primitive directions come first (those with
gcd(|dr|,|dc|) = 1), each group by increasing length — the order the
hand-written tables of 1.0.0 used, so 8, 16, 24 and 32 still produce exactly the
offsets they used to and old results stay comparable.

The non-primitive directions that appear from 24 upwards — (2,0) is two steps of
(1,0) — add no new heading, but they are not redundant: the cost of the long
move is computed from the slope over the *whole span*, so it can cross a dip
that the two short moves would have to climb into. More neighbours buy
smoothness at the price of ignoring what lies between the endpoints.

### 5.3 The cost of one move (`src/costfunctions.h`)

For an edge from *v* to *u*:

```
dz   = dem[u] - dem[v]                     (metres)
dh   = horizontal distance of the move     (metres, precomputed per offset)
sf   = dz / dh                             (slope as a fraction, not degrees)
```

`sf` is fed to the chosen cost function. Trajecta ships six:

| Cost function | What it returns | Unit of the result |
|---|---|---|
| Tobler's hiking function | walking speed from slope | hours |
| Modified Tobler (White 2015) | idem, recalibrated | hours |
| Márquez-Pérez et al. (2017) | idem, recalibrated | hours |
| Herzog (2013) | metabolic cost | kilojoules per kilogram |
| Campbell (2019), 5th and 50th percentile | speed from a large GPS corpus | hours |

The choice therefore fixes what the numbers *are*: a Herzog result and a Tobler
result are not two versions of the same map, they answer different questions
(energy against time), and their densities cannot be compared.

Three implementation notes:

- **No `sqrt` in the inner loop.** `dh` and `1/dh` are precomputed per offset,
  once, before the search starts. The inner loop of a FETE run executes this
  arithmetic on the order of 10⁸ times per source point.
- **No `exp` either, by default.** The engine builds a 65 536-entry lookup table
  over the plausible range of `sf` and interpolates linearly between entries.
  `TRAJECTA_EXACT_COST=1` in the environment turns the table off and calls the
  real function, which is how the table was verified.
- **The slope cut-off is a height comparison.** "Refuse moves steeper than 30°"
  is stored as a `dz` limit per offset, so the test is `dz > dz_up[k]` — no
  division, and perfectly branch-predicted when the option is off, which it is
  by default.

### 5.4 Cost modifiers, and barriers

Cost modifiers are a multiplier per cell, assembled before the search:

1. a vector layer carrying a `cost` attribute is rasterised, with a buffer of
   *n* cells per side, into a multiplier surface;
2. a multiplier raster aligned with the DEM is combined with it;
3. the product is what the search reads — `edge_cost *= multiplier[u]`.

*Treat extreme multipliers as impassable barriers* is not cosmetic. With a
multiplier of, say, 999 999 the cells are not impassable, merely very expensive,
so Dijkstra keeps expanding to enormous cost levels and settles the **entire**
raster for every source before early termination can fire. Marking those cells
impassable instead keeps each search tight, and matches what the user meant.

Impassable cells — NoData, barriers, anything outside the DEM — live in a
`passable` mask that the inner loop checks before anything else.

### 5.5 FETE: the propagation loop (`src/main_fete.cpp`)

FETE asks: *if everyone travelled between all of these places by the cheapest
route, which ground would be worn?* So for each sample point it runs a Dijkstra
over the whole grid and counts, for every cell, how many of those optimal routes
pass through it.

The loop over source points is parallel (`#pragma omp for schedule(dynamic)`);
each thread owns its own buffers. For one source:

**1 — Dijkstra.** A binary heap over `(cost, node)` pairs kept in a plain
`std::vector` with `push_heap`/`pop_heap`, rather than `std::priority_queue`,
so the container can be cleared and reused between sources without
reallocating. Settled nodes are recorded in `visit_order`; every node whose cost
was ever lowered is recorded in `touched`.

**2 — Early termination.** The search stops as soon as every sample point has
been settled (`dest_remaining` reaches zero), not when the heap empties. On a
large DEM with points clustered in one corner this is the difference between
minutes and hours.

**3 — Path counting, Brandes-style.** The naive way to count how many optimal
routes cross a cell is to walk the predecessor chain back from each of the P−1
destinations: P−1 random walks through memory, per source. Instead the engine
does one linear sweep. Each destination seeds `path_count[dest] += 1`, then
`visit_order` is walked **backwards** — far to near — and each node adds its
count to its predecessor:

```
for i from visit_count-1 down to 0:
    v    = visit_order[i]
    pred = predecessor[v]
    if pred >= 0: path_count[pred] += path_count[v]
```

Because `visit_order` is the order in which Dijkstra settled the nodes, walking
it backwards guarantees that a node's own count is complete before it is added
to its predecessor's. This is the accumulation step of Brandes' betweenness
algorithm, specialised to a single source and unique shortest paths.

**4 — Writing the density.** `visit_order` is swept once more and each non-zero
count is added to the shared density raster with `#pragma omp atomic`. There is
no per-thread copy of the raster to merge afterwards — on a 2000×2600 grid that
merge was 25 MB per thread and an O(N) pass per source.

**5 — Smart reset.** Between sources, only the cells in `touched` are reset,
not the whole grid. A search that settles 2 % of a 5-million-cell raster touches
100 000 cells; resetting all five million instead would dominate the run.

The density is written as counts of routes per cell. *Path smoothing* is applied
at the very end as a **separable box filter** over the finished density —
horizontal pass, then vertical — rather than by dilating around every cell as
it is written, which is what 1.0.0 did and what made the buffer expensive.

### 5.6 Checkpointing the propagation (`src/checkpoint.h`)

A FETE over a large DEM runs for days, and the only state the loop accumulates
is the density raster: every other array is either an input or is rebuilt
deterministically from the inputs in seconds. So a checkpoint is exactly two
things — how many sources are finished, and the density so far.

The loop is therefore run in **blocks**. At the end of a block every thread has
finished every source in it (the barrier at the end of an `omp for` guarantees
it), which is the only moment at which "sources 0..n−1 are complete" is a true
statement about a dynamically scheduled loop. Inside a block nothing changes.

Writing is timer-gated rather than block-gated: on a small DEM a block is
seconds and there is nothing worth saving; on a large one a block can be an
hour. Two files are kept and written alternately — a new one is written in full
under a temporary name and renamed into place, and only then is the older
removed. Rename is atomic on NTFS and on POSIX, so a machine that loses power
mid-write always leaves at least one complete checkpoint behind.

A **fingerprint** records what has to match for a resumed run to produce the
same numbers as an uninterrupted one: the sizes and modification times of the
DEM, the points and the modifier layers, the grid dimensions, the neighbourhood,
the cost function and the slope cut-off. If any of it differs the checkpoint is
refused rather than misread.

### 5.7 LCPA: one origin, chosen destinations (`src/main_lcpa.cpp`)

Same graph, same cost, one search: Dijkstra from the origin, stopping when every
destination is settled. Each route is then recovered by walking the predecessor
chain back from the destination — the only place where that walk is the right
tool, because here there is one chain per destination and not P−1 of them.

Routes are written twice: as a raster, and as lines carrying the total cost and
the length of each.

**The cost corridor** answers the question a single line cannot. For every cell:

```
excess(c) = ( cost(origin → c) + cost(c → destination) − best ) / best × 100
```

Zero on the optimal route itself; small wherever a detour through that cell
would cost almost nothing extra. Cells above the chosen percentage are written
as NoData, so a GIS draws nothing there instead of stretching its palette across
the whole map.

The second term is why the graph being directed matters: `cost(c → destination)`
is a different question from `cost(destination → c)`, so the engine runs a
**reversed** Dijkstra from each destination — following edges backwards, with
each edge's cost computed in its own direction — and adds the two surfaces.

A narrow corridor means the terrain dictated the route. A wide one means the
line on the map is one of many nearly equal options, and should not be read as
*the* route. That is the honest form of a least-cost path result, and it is the
reason the option exists.

### 5.8 What the memory estimate is for

Before the propagation starts, the engine computes what the run will need:
17 bytes per cell shared (DEM, passable mask, multipliers, density) plus about
23 bytes per cell **per thread** (cumulative cost, predecessor, visited flag,
visit order, touched list, path counts, heap). On a 5-million-cell raster with
16 threads that is roughly 1.9 GB — which is why the number of threads is a
memory decision as much as a speed one, and why the form asks for a RAM ceiling
and warns when the two do not fit together.
