Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
root = fs.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
shell.Run Chr(34) & root & "\python\pythonw.exe" & Chr(34) & " -m src.phase_j.distribution_app --root " & Chr(34) & root & Chr(34), 0, False
