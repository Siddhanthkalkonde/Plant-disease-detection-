# YOLOv8 Web Detection with Wikipedia Integration 🌐

> *When you want to flex your YOLO skills but make it accessible to your non-technical friends* 💻

An absolutely fire web-based real-time object detection system that not only finds objects but also spills the tea about them using Wikipedia. This beast serves up live detection through your browser while dropping knowledge bombs about whatever it spots. Built different, hits different.

## Features That Go Absolutely Mental

- 🌐 **Web-based interface** - Because desktop apps are so 2010
- 📹 **Live streaming detection** - Two feeds: raw and annotated (we're fancy like that)
- 📸 **Image capture** - Click button, get results, feel smart
- 📚 **Wikipedia integration** - Auto-fetches info about detected objects
- 🎯 **Dual video feeds** - Raw stream vs annotated stream side by side
- 🖼️ **Image saving** - Timestamps and unique IDs because we're organized
- 📱 **Responsive design** - Works on your phone, tablet, whatever

## Requirements (The Essentials)

Don't even think about running this without installing these:

```bash
pip install ultralytics      # YOLO detection overlord
pip install opencv-python    # Computer vision OG
pip install supervision      # Detection visualization boss
pip install numpy            # Math libraries go brrrr
pip install flask            # Web framework that doesn't suck
pip install wikipedia        # Knowledge database API
pip install requests         # HTTP requests handler
```

## Usage (Time to Get This Show Started)

### Launch the Web Server

```bash
# Basic launch (localhost vibes)
python your_script.py

# Custom resolution because you're extra
python your_script.py --webcam-resolution 1920 1080

# Custom port and host (for the networking pros)
python your_script.py --port 8080 --host 127.0.0.1

# Video file instead of webcam
python your_script.py --input /path/to/your/epic/video.mp4

# Custom save directory for captured images
python your_script.py --save-dir my_awesome_captures
```

### Command Line Arguments (The Power User Menu)

| Argument | What It Does | Default | Example |
|----------|-------------|---------|---------|
| `--webcam-resolution` | Camera quality settings | [1280, 720] | `--webcam-resolution 640 480` |
| `--input` | Video source (webcam/file) | "webcam" | `--input my_video.mp4` |
| `--port` | Web server port | 5000 | `--port 8080` |
| `--host` | Server host address | "0.0.0.0" | `--host 127.0.0.1` |
| `--save-dir` | Where to save captures | "captured_images" | `--save-dir cool_pics` |

## How This Absolute Unit Works

1. **Video Processing Thread** 🎬: Grabs frames from camera/video in background
2. **Flask Web Server** 🌐: Serves the slick web interface
3. **Live Streaming** 📹: Two video feeds - raw and processed
4. **Capture & Inference** 📸: Click button → capture → detect → annotate
5. **Wikipedia Magic** 📚: Auto-fetches info about detected objects
6. **Real-time Display** ✨: Shows everything in your browser like a boss

## Web Interface Features

### What You'll Actually See

- **Dual video streams**: Raw feed on left, processed on right
- **Big capture button**: Click it, things happen
- **Status updates**: Real-time feedback that doesn't lie
- **Object information**: Wikipedia summaries for detected objects
- **Responsive design**: Looks good on any device

### The User Experience

1. **Load the page** - Clean, professional interface loads
2. **Watch live streams** - Two feeds showing your camera/video
3. **Hit capture** - Button processes current frame
4. **Get results** - Annotated image with detected objects
5. **Read about objects** - Wikipedia info auto-loads
6. **Feel smart** - Knowledge acquired, mission accomplished

## Model Setup (The Foundation)

Update this path or prepare for disappointment:

```python
model_path = "/home/sid/trial/trained models/2-model disease .pt"
```

Make sure your model file actually exists and isn't corrupted.

## Directory Structure (Stay Organized)

The app creates these automatically:

```
your_project/
├── templates/
│   └── index.html          # Auto-generated web interface
├── captured_images/        # Your saved images go here
│   ├── 20241215_143052_a1b2c3.jpg
│   └── 20241215_143127_d4e5f6.jpg
└── your_script.py         # This magnificent code
```

## API Endpoints (For the Developers)

- `GET /` - Main web interface
- `GET /video_feed` - Raw video stream
- `GET /inference_feed` - Annotated video stream  
- `POST /capture` - Capture image and run inference
- `GET /object_info` - Get current object information

## Customization Options (Make It Yours)

### Change the Model
```python
model_path = "/path/to/your/custom/model.pt"
```

### Adjust Video Quality
```python
# In process_frames function
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
```

### Modify Wikipedia Searches
```python
# In get_object_info function
summary = wikipedia.summary(page_title, sentences=5)  # More detailed info
```

## Troubleshooting (When Reality Hits)

**Web page won't load?**
- Check if the port is already in use
- Try a different port with `--port 8080`

**Video feed not showing?**
- Camera might be in use by another app
- Try different camera index: `cv2.VideoCapture(1)`

**Wikipedia info not loading?**
- Check internet connection
- Some objects don't have Wikipedia pages (shocking, we know)

**Model not loading?**
- Verify the model path exists
- Make sure it's a valid YOLOv8 model file

**Capture button not working?**
- Check browser console for JavaScript errors
- Try refreshing the page

## Performance Tips (Keep It Smooth)

- **Use reasonable resolution** - 1280x720 is usually plenty
- **Close unnecessary browser tabs** - They eat RAM like crazy  
- **Check your model size** - Smaller models = faster inference
- **Monitor CPU usage** - This thing can be hungry

## Security Notes (Don't Get Hacked)

- **Don't expose to public internet** without proper security
- **Use localhost** for development only
- **Be careful with file paths** in production
- **Validate inputs** if you modify the code

## Fun Features to Add

- **Object counting statistics**
- **Detection confidence filtering**  
- **Multiple model support**
- **Custom annotation colors**
- **Export detection results as JSON**

---

**Built by someone who thinks web interfaces are better than terminal outputs** 🌐💻

*P.S. - If Wikipedia starts rate-limiting you, that means you're using this too much. Take a break.*

*Last updated: When we remembered to add timestamps to everything*
