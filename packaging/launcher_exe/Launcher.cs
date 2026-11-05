// MIT License。起動前の処理は既存の Python 側へ委譲する。
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

internal static class Launcher
{
#if UNINSTALL
    private const bool Uninstall = true;
#else
    private const bool Uninstall = false;
#endif
    private const int Failure = 1;
    private const int PrimaryLanguageMask = 0x3ff;
    private const int JapaneseLanguage = 0x11;
    [DllImport("kernel32.dll")]
    private static extern ushort GetUserDefaultUILanguage();
    private const string Bootstrap = "import sys;from pathlib import Path;"
        + "root=Path(sys.argv[1]);sys.path.insert(0,str(root/'tools'/'launcher'));"
        + "from signed_launcher import main;"
        + "sys.exit(main(root,sys.argv[2:],False))";

    // Windows の argv 規則に従い、引用符の前と末尾の逆斜線を二重化する。
    internal static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char ch in value)
        {
            if (ch == '\\') { slashes++; continue; }
            result.Append('\\', ch == '"' ? slashes * 2 + 1 : slashes);
            result.Append(ch);
            slashes = 0;
        }
        result.Append('\\', slashes * 2);
        return result.Append('"').ToString();
    }

    [STAThread]
    private static int Main(string[] args)
    {
        string root = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        string python = Path.Combine(root, "python", Uninstall ? "python.exe" : "pythonw.exe");
        if (!File.Exists(python) || !File.Exists(Path.Combine(root, "python", "python312.dll")))
        {
            MessageBox.Show("ZIP をすべて展開してから開いてください。\n"
                + "Please extract all files from the ZIP before opening.", "puyoZeus Live",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            return Failure;
        }
        try
        {
            return RunPython(root, python, args);
        }
        catch (Exception error)
        {
            if (Uninstall) Console.WriteLine(error.Message);
            else MessageBox.Show("起動できませんでした / Unable to start\n" + error.Message,
                    "puyoZeus Live", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return Failure;
        }
        finally
        {
            if (Uninstall)
            {
                Console.Write((GetUserDefaultUILanguage() & PrimaryLanguageMask) == JapaneseLanguage
                    ? "Enter で閉じます。" : "Press Enter to close.");
                try { Console.ReadLine(); }
                catch (IOException) { }
            }
        }
    }

    private static int RunPython(string root, string python, string[] args)
    {
        string command;
        if (Uninstall)
        {
            // 停止確認が親の Python を誤検出しないよう、一段で実行する。
            command = "-s -m src.phase_j.distribution_uninstall --root "
                + Quote(root) + " --interactive";
        }
        else
        {
            StringBuilder launch = new StringBuilder("-s -c ");
            launch.Append(Quote(Bootstrap)).Append(' ').Append(Quote(root));
            foreach (string arg in args) launch.Append(' ').Append(Quote(arg));
            command = launch.ToString();
        }
        ProcessStartInfo start = new ProcessStartInfo(python, command);
        start.WorkingDirectory = root;
        start.UseShellExecute = false;
        start.CreateNoWindow = !Uninstall;
        using (Process child = Process.Start(start))
        {
            child.WaitForExit();
            return child.ExitCode;
        }
    }
}
