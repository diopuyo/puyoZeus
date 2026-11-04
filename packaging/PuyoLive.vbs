Set shell = CreateObject("WScript.Shell")
Set fs = CreateObject("Scripting.FileSystemObject")
root = fs.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = root
arguments = ""
For Each argument In WScript.Arguments
    If argument = "--autostart" Then arguments = " --autostart"
Next
shell.Run "python\pythonw.exe -s -m src.phase_j.distribution_app --root ." & arguments, 0, False
