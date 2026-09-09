using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Windows.Forms;

internal static class Launcher
{
    private const string AppUrl = "http://127.0.0.1:5173";
    private const string HealthUrl = "http://127.0.0.1:8000/api/health";

    [STAThread]
    private static int Main()
    {
        Application.EnableVisualStyles();

        try
        {
            var root = Path.GetFullPath(AppDomain.CurrentDomain.BaseDirectory);
            var script = Path.Combine(root, "verifyvision.ps1");
            if (!File.Exists(script))
            {
                ShowError("verifyvision.ps1 was not found beside VerifyVision.exe.");
                return 1;
            }

            if (!IsReady())
            {
                string output;
                var exitCode = RunPowerShell(root, script, out output);
                if (exitCode != 0)
                {
                    ShowError(GetFailureMessage(output));
                    return 1;
                }
                if (!IsReady())
                {
                    ShowError("VerifyVision started, but the app is not reachable.");
                    return 1;
                }
            }

            Process.Start(new ProcessStartInfo(AppUrl) { UseShellExecute = true });
            return 0;
        }
        catch (Exception error)
        {
            ShowError(error.Message);
            return 1;
        }
    }

    private static bool IsReady()
    {
        return IsReachable(HealthUrl) && IsReachable(AppUrl);
    }

    private static bool IsReachable(string url)
    {
        try
        {
            var request = (HttpWebRequest)WebRequest.Create(url);
            request.Method = "GET";
            request.Proxy = null;
            request.Timeout = 750;
            request.ReadWriteTimeout = 750;
            request.KeepAlive = false;
            using (var response = (HttpWebResponse)request.GetResponse())
            {
                return response.StatusCode == HttpStatusCode.OK;
            }
        }
        catch (WebException)
        {
            return false;
        }
    }

    private static int RunPowerShell(string root, string script, out string output)
    {
        var powerShell = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.System),
            @"WindowsPowerShell\v1.0\powershell.exe"
        );
        var startInfo = new ProcessStartInfo
        {
            FileName = powerShell,
            Arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File \"" + script + "\" start",
            WorkingDirectory = root,
            UseShellExecute = false,
            CreateNoWindow = true,
            WindowStyle = ProcessWindowStyle.Hidden,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };

        using (var process = Process.Start(startInfo))
        {
            if (process == null)
            {
                throw new InvalidOperationException("PowerShell could not start.");
            }
            var standardOutput = process.StandardOutput.ReadToEnd();
            var standardError = process.StandardError.ReadToEnd();
            process.WaitForExit();
            output = (standardError + Environment.NewLine + standardOutput).Trim();
            return process.ExitCode;
        }
    }

    private static string GetFailureMessage(string output)
    {
        if (output.IndexOf("not set up yet", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return "VerifyVision is not set up yet. Run \".\\verifyvision.ps1 all\" once, then try again.";
        }

        var markers = new[]
        {
            "Port ",
            "LocateAnything CUDA runtime unavailable",
            "Backend exited",
            "Frontend exited",
            "did not become",
        };
        foreach (var line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
        {
            foreach (var marker in markers)
            {
                if (line.IndexOf(marker, StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    var reason = line.Trim();
                    return reason.Length <= 240 ? reason : reason.Substring(0, 240);
                }
            }
        }
        return "VerifyVision could not start.";
    }

    private static void ShowError(string reason)
    {
        MessageBox.Show(
            reason + Environment.NewLine + Environment.NewLine
                + "Run \".\\verifyvision.ps1 diagnose\" for more information.",
            "VerifyVision",
            MessageBoxButtons.OK,
            MessageBoxIcon.Error
        );
    }
}
