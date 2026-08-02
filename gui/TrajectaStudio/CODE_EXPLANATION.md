# Trajecta Studio — Complete explanation (general logic and code)

This document explains **how the Trajecta graphical interface works**, first
at the architecture level and then diving into the code file by file,
block by block.

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

Four pages in a `QStackedWidget`, navigated from a sidebar:

1. **Analysis setup** — the complete form: mode (FETE/LCPA, two selectable
   cards), input data (rows that change with the mode), cost modifiers
   (checkable group), algorithm, performance (defaults computed from the
   real hardware), output file names, Run button.
2. **Run & results** — status chip, progress bar, elapsed time, ANSI-aware
   live console, summary card, Cancel / Open folder buttons.
3. **Guide** — built-in user manual in HTML (modes, file requirements,
   parameters, GDAL, tips).
4. **About** — version, author, license, citations.

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
as the engine's CMake).

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
5. **Performance**: defaults computed from the hardware as the engine does:
   threads = `idealThreadCount()-4` (min 1), RAM = 60% of physical. The hint
   labels show the detected values. Verbose checkbox.
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
missing) → `saveSettings()` → Run page reset (clean console, "Launching…"
marker, bar at 0, RUNNING chip, cancel enabled, run disabled, stopwatch
started) → `switchPage(1)` → `m_runner->start()`.

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
