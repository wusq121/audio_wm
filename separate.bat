ffmpeg -i ./combine/bat-5min.mkv -vn -ar 44100 -ac 2 -f wav ./embedded/batman-5min8192.wav
ffmpeg -i ./combine/bat-15min.mkv -vn -ar 44100 -ac 2 -f wav ./embedded/batman-15min8192.wav
ffmpeg -i ./combine/bat-30min.mkv -vn -ar 44100 -ac 2 -f wav ./embedded/batman-30min8192.wav
