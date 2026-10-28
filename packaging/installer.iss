; Inno Setup script for a Windows installer (run after PyInstaller).
;   iscc packaging\installer.iss   (optionally /DAppVersion=1.2.3)
#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif

[Setup]
AppName=LookSay
AppVersion={#AppVersion}
AppPublisher=LookSay
DefaultDirName={autopf}\LookSay
DefaultGroupName=LookSay
OutputDir=..\dist
OutputBaseFilename=LookSay-{#AppVersion}-setup
SetupIconFile=..\assets\logo.ico
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
Source: "..\dist\LookSay\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\LookSay"; Filename: "{app}\LookSay.exe"
Name: "{autodesktop}\LookSay"; Filename: "{app}\LookSay.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"

[Run]
Filename: "{app}\LookSay.exe"; Description: "Start LookSay"; Flags: nowait postinstall skipifsilent
