import vlc

# Create an instance of the VLC player
instance = vlc.Instance()

# Create a media player object
player = instance.media_player_new()

# Specify the camera stream URL (replace with your actual URL)
stream_url = "rtsp://admin:1adctester@172.30.3.184:554/s1"

# Create a media object from the stream URL
media = instance.media_new(stream_url)

# Set the media player's media
player.set_media(media)

# Play the stream
player.play()

# Keep the script running until you want to stop the stream
while True:
    pass
