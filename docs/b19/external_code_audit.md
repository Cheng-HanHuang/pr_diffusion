]633;E;{   echo "# B19 external code audit"\x3b   echo\x3b   date\x3b   echo\x3b   echo "## Submodule status"\x3b   git submodule status\x3b   echo\x3b   echo "## License files"\x3b   find external -maxdepth 3 \\( -iname "LICENSE*" -o -iname "COPYING*" -o -iname "NOTICE*" \\) -print | sort\x3b } > docs/b19/external_code_audit.md;7a78a9fe-23b1-4c8d-93c9-027294aa28e9]633;C# B19 external code audit

Tue Jun 23 10:19:58 AM EDT 2026

## Submodule status
 e7a77d094167084faed19b599b96673b7bb11447 external/daps (heads/main)
 e3101e7e0c16d63bc904a641a26aa8b87de62d0a external/dmplug (heads/main)
 effbde7325b22ce8dc3e2c06c160c021e743a12d external/dps (heads/main)
 d5bfd15f580432c31c9aa7ba5a1b20d3b37edb17 external/mcg_diffusion (heads/main)
 275ab67efbd8146bffca20155171ba6be1169c09 external/sitcom_ode (heads/main)
 52f2c37e587576d02e2b27ac971e247f2899fc5e external/sitcom_ode_npsitcom (heads/main)

## License files
external/daps/LICENSE
external/mcg_diffusion/LICENSE
