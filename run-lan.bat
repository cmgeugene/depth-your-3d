@echo off
rem Depth Studio - share on the internal network (http://192.168.x.x:8000).
rem Listens on every address of this PC; the address to share is printed at startup.
rem Keep this file ASCII: cmd misreads batch files with UTF-8 text after chcp 65001.
set DS_HOST=0.0.0.0
call "%~dp0run.bat"
