@echo off
setlocal

set BASE_URL=http://127.0.0.1:8001
set IMAGE_PATH=D:\漫画\未翻译\1\054.jpg
set CONFIG_PATH=D:\漫画\12\examples\config.json
set OUT_PATH=D:\漫画\12\result_api.png

curl.exe -s -S -X POST "%BASE_URL%/translate/with-form/image" ^
  -F "image=@%IMAGE_PATH%" ^
  -F "config=<%CONFIG_PATH%" ^
  --connect-timeout 10 ^
  --max-time 600 ^
  --output "%OUT_PATH%"

echo Saved: %OUT_PATH%
pause
