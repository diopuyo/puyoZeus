; 管理者権限を要求しない、署名なしの試作配布版。
#ifndef ReleaseVersion
  #define ReleaseVersion "0.6.0-preview.11"
#endif
#ifndef ReleaseCodeSHA
  #define ReleaseCodeSHA "unknown"
#endif
#ifndef ReleaseModelSHA
  #define ReleaseModelSHA "unknown"
#endif
[Setup]
AppId=PuyoLive-EarlyAccess-Distribution
AppName={cm:BrandName}
AppVerName={cm:BrandName} {#ReleaseVersion}
AppVersion={#ReleaseVersion}
DefaultDirName={localappdata}\Programs\PuyoLive-EarlyAccess
DefaultGroupName=puyoZeus Live
PrivilegesRequired=lowest
Uninstallable=yes
UsePreviousAppDir=yes
DisableDirPage=no
DisableProgramGroupPage=yes
SetupLogging=yes
OutputDir={#ReleaseDir}
OutputBaseFilename=puyoZeus-Live-{#ReleaseVersion}-setup
VersionInfoDescription=puyoZeus Live Setup
VersionInfoProductName=puyoZeus Live
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
SetupIconFile=..\src\phase_j\brand\app_logo.ico
WizardImageFile=..\src\phase_j\brand\wizard_logo.bmp
WizardSmallImageFile=..\src\phase_j\brand\wizard_small.bmp
UninstallDisplayIcon={app}\app\src\phase_j\brand\app_logo.ico
CloseApplications=no
RestartApplications=no
LicenseFile=..\LICENSE
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={cm:BrandName}

[Languages]
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
ja.BrandName=puyoZeus Live
en.BrandName=puyoZeus Live
ja.InitialSetup=puyoZeus Live 初期設定
en.InitialSetup=puyoZeus Live Initial setup
ja.Uninstall=puyoZeus Live をアンインストール
en.Uninstall=Uninstall puyoZeus Live

[Icons]
Name: "{group}\{cm:Uninstall}"; Filename: "{uninstallexe}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "distribution.json,puyo_live.json,output\*,logs\*,install.log"
Source: "{#SourceDir}\tools\SendLogs.ps1"; Flags: dontcopy

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-s -m src.phase_j.distribution_app --root . --setup"; WorkingDir: "{app}"; Description: "{cm:InitialSetup}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
; 保存後の再インストールでは既存扱いになる親も、空の場合だけ片付ける。
Type: dirifempty; Name: "{app}\app\config"
Type: dirifempty; Name: "{app}\app"
Type: dirifempty; Name: "{app}"

[Code]
var
  OptionsPage: TInputOptionWizardPage;
  PortPage: TInputQueryWizardPage;
  InstallCompleted: Boolean;
  DeleteUserData: Boolean;

function UninstallProcessId: LongWord;
  external 'GetCurrentProcessId@kernel32.dll stdcall';

function L(Ja, En: String): String;
begin
  if ActiveLanguage = 'ja' then Result := Ja else Result := En;
end;

function CleanupCommand(Options: String): Boolean;
var ExitCode: Integer;
begin
  Result := Exec(ExpandConstant('{app}\python\python.exe'),
    '-s -B -m src.phase_j.distribution_uninstall --root . ' + Options,
    ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ExitCode);
  Result := Result and (ExitCode = 0);
end;

function ScheduleTempCleanup: Boolean;
var TempDirectory, Helper, Args: String; ExitCode: Integer;
begin
  Result := False;
  if CompareText(ExtractFileName(ParamStr(0)), '_unins.tmp') <> 0 then Exit;
  TempDirectory := ExtractFileDir(ParamStr(0));
  Helper := TempDirectory + '\PuyoLiveUninstallCleanup.ps1';
  if not FileCopy(ExpandConstant('{app}\tools\UninstallCleanup.ps1'), Helper, False) then Exit;
  Args := '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' +
    Helper + '" -UninstallerPid ' + IntToStr(UninstallProcessId);
  Result := Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Args,
    ExpandConstant('{win}'), SW_HIDE, ewNoWait, ExitCode);
  if Result then Log('puyoZeus temporary cleanup: ' + TempDirectory)
  else DeleteFile(Helper);
end;

function InitializeUninstall: Boolean;
begin
  Result := CleanupCommand('--check');
  if not Result then begin
    if not UninstallSilent then MsgBox(L(
      'puyoZeus Live の設定で「アプリを終了」を押してから再実行してください。停止確認に失敗した場合も削除しません。',
      'Exit puyoZeus Live from Settings, then retry. Nothing is removed if the stop check fails.'), mbError, MB_OK);
    Exit;
  end;
  DeleteUserData := False;
  if UninstallSilent then
    DeleteUserData := ExpandConstant('{param:DELETEUSERDATA|0}') = '1'
  else
    DeleteUserData := MsgBox(L(
      '設定・ログ・記録 (output/、logs/、distribution.json、機器較正 等) も削除しますか？「いいえ」で残します。',
      'Delete settings, logs and recordings (output/, logs/, distribution.json, device calibration, etc.) too? Choose No to keep them.'),
      mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var Options: String;
begin
  if CurUninstallStep = usUninstall then begin
    if not ScheduleTempCleanup then
      RaiseException(L('一時ファイルの後片付けを開始できませんでした。再実行してください。',
        'Could not start temporary-file cleanup. Please retry.'));
    Options := '';
    if DeleteUserData then Options := '--delete-data';
    if not CleanupCommand(Options) then
      RaiseException(L('停止・後片付けに失敗しました。アプリを終了して再実行してください。',
        'Stop/cleanup failed. Exit the app and retry.'));
  end;
end;

procedure InitializeWizard;
begin
  ExtractTemporaryFile('SendLogs.ps1');
  OptionsPage := CreateInputOptionPage(wpSelectDir, L('起動と表示', 'Startup and display'),
    L('初期設定を選びます', 'Choose initial settings'),
    L('インストール後、入力の自動検出・テスト、見本付きの配置とテーマ、OBSのURLコピーを同じ初期設定画面で行います。',
      'After installation, the setup page detects and tests inputs, previews layouts and themes, and copies your OBS URL.'), False, False);
  OptionsPage.Add(L('マスコットを表示', 'Show mascot'));
  OptionsPage.Add(L('デスクトップにショートカット', 'Desktop shortcut'));
  OptionsPage.Add(L('スタートメニューにショートカット', 'Start menu shortcut'));
  OptionsPage.Add(L('Windows起動時に自動起動', 'Start with Windows'));
  OptionsPage.Add(L('更新を確認（通知だけ。自動でインストールしません）', 'Check updates (notify only; no automatic installation)'));
  OptionsPage.Add(L('詳細設定を開く（ポート）', 'Open advanced settings (port)'));
  OptionsPage.Values[2] := True;
  OptionsPage.Values[4] := True;
  PortPage := CreateInputQueryPage(OptionsPage.ID, L('詳細設定', 'Advanced settings'),
    L('通常は変更不要です', 'The default normally works'),
    L('使用中なら自動で空きポートを選びます。障害ログは自動送信しません。',
      'A free port is selected if busy. Logs are never sent automatically.'));
  PortPage.Add('Port:', False);
  PortPage.Values[0] := '8765';
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = PortPage.ID) and not OptionsPage.Values[5];
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var Port: Integer;
begin
  Result := True;
  if CurPageID = PortPage.ID then begin
    Port := StrToIntDef(PortPage.Values[0], 0);
    Result := (Port >= 1024) and (Port <= 65535);
    if not Result then MsgBox('Port: 1024-65535', mbError, MB_OK);
  end;
end;

function B(Value: Boolean): String;
begin
  if Value then Result := 'true' else Result := 'false';
end;

procedure CurStepChanged(CurStep: TSetupStep);
var Config: String;
begin
  if CurStep = ssPostInstall then begin
    Config := '{"completed":false,"language":"' + ActiveLanguage + '","port":' + PortPage.Values[0] +
      ',"overlay":{"mascot":' + B(OptionsPage.Values[0]) + '},"desktop_shortcut":' + B(OptionsPage.Values[1]) +
      ',"start_menu_shortcut":' + B(OptionsPage.Values[2]) + ',"autostart":' + B(OptionsPage.Values[3]) +
      ',"check_updates":' + B(OptionsPage.Values[4]) + '}';
    if not FileExists(ExpandConstant('{app}\distribution.json')) then
      SaveStringToFile(ExpandConstant('{app}\distribution.json'), Config, False);
    FileCopy(ExpandConstant('{log}'), ExpandConstant('{app}\install.log'), False);
    InstallCompleted := True;
    WizardForm.FinishedLabel.Caption := L(
      '次の初期設定画面で入力を選び、見本を確認して起動してください。OBS仮想カメラは「ソース」出力、ブラウザソースは1920×1080。最後に実際の空きポートのURLをコピーできます。非公式のファン制作ツールです。',
      'Open initial setup to select an input, preview the overlay and start. Set OBS Virtual Camera to Source output and Browser source to 1920x1080. Copy the actual free-port URL there. This is an unofficial fan-made tool.');
  end;
end;

procedure DeinitializeSetup;
var ExitCode: Integer; Args: String;
begin
  if not InstallCompleted and FileExists(ExpandConstant('{tmp}\SendLogs.ps1')) then begin
    if MsgBox(L('インストールが完了しませんでした。「ログを送る」の確認画面を開きますか？内容を表示し、SEND と入力した時だけ送信します。',
      'Installation did not finish. Open Send logs? The report is shown first and sent only when you type SEND.'), mbConfirmation, MB_YESNO) = IDYES then begin
      Args := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{tmp}\SendLogs.ps1') + '" -LogPath "' + ExpandConstant('{log}') + '" -AppVersion "{#ReleaseVersion}" -CodeSha "{#ReleaseCodeSHA}" -ModelSha "{#ReleaseModelSHA}"';
      Exec(ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'), Args, '', SW_SHOWNORMAL, ewWaitUntilTerminated, ExitCode);
    end;
  end;
end;
