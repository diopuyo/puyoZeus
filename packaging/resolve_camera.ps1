# Resolve the DirectShow index of a named video device (default: OBS Virtual Camera)
# and write it to puyo_live.json "device_index". OpenCV CAP_DSHOW uses the same enumeration order.
# Usage: resolve_camera.ps1 [-Name "OBS Virtual Camera"] [-ListOnly]
param([string]$Name = 'OBS Virtual Camera', [switch]$ListOnly)
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
[ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface ICreateDevEnum {
  [PreserveSig] int CreateClassEnumerator([In] ref Guid clsid, out IEnumMoniker em, int flags);
}
[ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
interface IPropertyBag {
  [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string name, out object val, IntPtr log);
  [PreserveSig] int Write([MarshalAs(UnmanagedType.LPWStr)] string name, ref object val);
}
public static class DsVideoInputs {
  public static string[] Names() {
    var result = new List<string>();
    Guid sysEnum = new Guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86");
    Guid videoCat = new Guid("860BB310-5D01-11d0-BD3B-00A0C911CE86");
    var dev = (ICreateDevEnum)Activator.CreateInstance(Type.GetTypeFromCLSID(sysEnum));
    IEnumMoniker em;
    if (dev.CreateClassEnumerator(ref videoCat, out em, 0) != 0 || em == null) return result.ToArray();
    var m = new IMoniker[1];
    Guid bagId = typeof(IPropertyBag).GUID;
    while (em.Next(1, m, IntPtr.Zero) == 0) {
      object bagObj;
      m[0].BindToStorage(null, null, ref bagId, out bagObj);
      object val;
      ((IPropertyBag)bagObj).Read("FriendlyName", out val, IntPtr.Zero);
      result.Add(val as string);
    }
    return result.ToArray();
  }
}
'@
$names = [DsVideoInputs]::Names()
for ($i = 0; $i -lt $names.Count; $i++) { Write-Host ("index {0}: {1}" -f $i, $names[$i]) }
if ($ListOnly) { exit 0 }
$hits = @(for ($i = 0; $i -lt $names.Count; $i++) { if ($names[$i] -eq $Name) { $i } })
if ($hits.Count -ne 1) {
  Write-Host "ERROR: '$Name' found $($hits.Count) times. Start OBS Virtual Camera (or check List_Cameras.cmd) and retry."
  exit 1
}
$json = Join-Path $PSScriptRoot 'puyo_live.json'
$text = [IO.File]::ReadAllText($json)
if ($text -notmatch '"device_index"\s*:\s*\d+') { Write-Host "ERROR: device_index not found in $json"; exit 1 }
$new = [regex]::Replace($text, '"device_index"\s*:\s*\d+', '"device_index": ' + $hits[0])
if ($new -ne $text) { [IO.File]::WriteAllText($json, $new, (New-Object Text.UTF8Encoding($false))) }
Write-Host "$Name -> device_index $($hits[0])"
exit 0
