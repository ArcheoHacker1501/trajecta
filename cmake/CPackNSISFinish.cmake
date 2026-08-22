# Read by CPack itself, once, just before the installer is generated.
#
# It exists for one line of NSIS, and the reason it cannot live in CMakeLists.txt
# is worth writing down: a CPACK_* variable set there is written into
# CPackConfig.cmake and parsed a second time, and neither backslashes nor double
# quotes survive that round trip. Doubling the backslashes gets them through but
# CPack then turns every quote into a semicolon, which produces an NSIS script
# that compiles and does nothing. A CPACK_PROJECT_CONFIG_FILE is parsed once,
# so what is written here is what reaches project.nsi.

if(CPACK_GENERATOR STREQUAL "NSIS")
  # "Run Trajecta Studio" on the finish page — started THROUGH EXPLORER, not by
  # the installer.
  #
  # The plain CPACK_NSIS_MUI_FINISHPAGE_RUN form makes NSIS Exec the program as
  # a child of the installer, and the installer is elevated because it writes to
  # Program Files. The application inherits that elevated token, and Windows
  # refuses to let a medium-integrity process — File Explorer — drag anything
  # into a high-integrity one (UIPI). Drag & drop was therefore dead on the
  # first run after every install and alive on the next one, with nothing in the
  # interface to explain the difference. It also left whatever that first run
  # wrote owned by the wrong account.
  #
  # explorer.exe is already running as the user, so handing it the path starts
  # the program with the user's own token. A function is needed because
  # MUI_FINISHPAGE_RUN can only name the executable itself; only
  # MUI_FINISHPAGE_RUN_FUNCTION can run something else on its behalf.
  set(CPACK_NSIS_INSTALLER_MUI_FINISHPAGE_RUN_CODE
"!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT \"Run Trajecta Studio\"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchTrajectaAsUser
Function LaunchTrajectaAsUser
  Exec '\"$WINDIR\\explorer.exe\" \"$INSTDIR\\TrajectaStudio.exe\"'
FunctionEnd
")
endif()
