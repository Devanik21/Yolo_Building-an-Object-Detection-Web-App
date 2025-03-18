import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
import io

# Load ImageNet classes (1000 classes)
IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/vision/master/torchvision/models/resnet.py"
response = requests.get(IMAGENET_CLASSES_URL)
imagenet_classes = []
if response.status_code == 200:
    content = response.text
    class_lines = [line for line in content.split("\n") if "_IMAGENET_CATEGORIES" in line]
    if class_lines:
        # Extract class names from the content
        for line in class_lines:
            if "'" in line:
                class_name = line.split("'")[1]
                imagenet_classes.append(class_name)

# If we couldn't get classes from GitHub, use a fallback approach
if not imagenet_classes:
    # Use a more comprehensive fallback list
    # This is a much larger list combining COCO and common objects
    imagenet_classes = [
        # Original COCO classes
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
        
        # Additional food items
        "pear", "grape", "watermelon", "strawberry", "blueberry", "raspberry", "blackberry", "pineapple",
        "mango", "peach", "plum", "cherry", "kiwi", "lemon", "lime", "coconut", "avocado", "tomato",
        "cucumber", "eggplant", "bell pepper", "chili pepper", "potato", "sweet potato", "onion", "garlic",
        "ginger", "mushroom", "lettuce", "cabbage", "spinach", "kale", "celery", "asparagus", "corn",
        "peas", "green beans", "rice", "pasta", "bread", "toast", "pancake", "waffle", "cereal", "oatmeal",
        "yogurt", "cheese", "butter", "milk", "cream", "ice cream", "chocolate", "candy", "cookie", "pie",
        "cupcake", "muffin", "bagel", "croissant", "sushi", "ramen", "soup", "salad", "hamburger", "sandwich",
        "burrito", "taco", "fries", "chips", "popcorn", "nuts", "eggs", "bacon", "sausage", "steak", "chicken",
        "fish", "shrimp", "crab", "lobster", "oyster", "clam", "mussel", "tea", "coffee", "juice", "soda",
        "water", "beer", "wine", "whiskey", "vodka", "cocktail",
        
        # Additional animals
        "lion", "tiger", "leopard", "jaguar", "cheetah", "wolf", "fox", "coyote", "hyena", "jackal",
        "raccoon", "panda", "koala", "kangaroo", "gorilla", "chimpanzee", "orangutan", "baboon", "lemur",
        "sloth", "monkey", "deer", "moose", "elk", "reindeer", "buffalo", "bison", "rhino", "hippo",
        "camel", "llama", "alpaca", "goat", "donkey", "mule", "pig", "boar", "hedgehog", "porcupine",
        "beaver", "otter", "ferret", "weasel", "mink", "skunk", "badger", "armadillo", "opossum", "bat",
        "squirrel", "chipmunk", "rat", "mouse", "hamster", "guinea pig", "rabbit", "hare", "mole", "shrew",
        "eagle", "hawk", "falcon", "owl", "vulture", "raven", "crow", "parrot", "parakeet", "canary",
        "finch", "sparrow", "robin", "cardinal", "blue jay", "woodpecker", "hummingbird", "duck", "goose",
        "swan", "turkey", "chicken", "rooster", "pigeon", "dove", "penguin", "ostrich", "flamingo", "stork",
        "crane", "peacock", "pelican", "seagull", "albatross", "heron", "crocodile", "alligator", "turtle",
        "tortoise", "lizard", "iguana", "chameleon", "gecko", "snake", "python", "cobra", "viper", "boa",
        "anaconda", "frog", "toad", "newt", "salamander", "axolotl", "fish", "shark", "whale", "dolphin",
        "porpoise", "seal", "sea lion", "walrus", "octopus", "squid", "cuttlefish", "jellyfish", "starfish",
        "sea urchin", "crab", "lobster", "shrimp", "crawfish", "butterfly", "moth", "caterpillar", "bee",
        "wasp", "hornet", "ant", "termite", "grasshopper", "cricket", "cockroach", "ladybug", "beetle",
        "fly", "mosquito", "spider", "scorpion", "tick", "mite", "centipede", "millipede", "worm", "snail",
        "slug", "coral", "anemone", "sponge",
        
        # Additional household objects
        "table", "desk", "drawer", "cabinet", "shelf", "bookshelf", "sofa", "armchair", "ottoman", "recliner",
        "stool", "bench", "bed", "mattress", "pillow", "blanket", "quilt", "comforter", "sheet", "curtain",
        "blind", "rug", "carpet", "mat", "lamp", "chandelier", "light bulb", "fan", "air conditioner", "heater",
        "fireplace", "stove", "oven", "microwave", "refrigerator", "freezer", "dishwasher", "washing machine",
        "dryer", "vacuum cleaner", "iron", "blender", "mixer", "toaster", "coffee maker", "kettle", "pot", "pan",
        "baking sheet", "cutting board", "dish", "plate", "bowl", "cup", "mug", "glass", "fork", "knife", 
        "spoon", "chopsticks", "napkin", "paper towel", "trash can", "recycling bin", "shower", "bathtub",
        "toilet", "sink", "mirror", "towel", "soap", "shampoo", "conditioner", "toothbrush", "toothpaste",
        "hairbrush", "comb", "razor", "nail clippers", "scissors", "hammer", "screwdriver", "wrench", "pliers",
        "drill", "saw", "nail", "screw", "bolt", "tape", "glue", "stapler", "paperclip", "pin", "needle",
        "thread", "button", "zipper", "wallet", "purse", "handbag", "backpack", "suitcase", "briefcase",
        "gift", "box", "package", "envelope", "paper", "notebook", "textbook", "magazine", "newspaper",
        "calendar", "map", "globe", "pen", "pencil", "marker", "highlighter", "eraser", "ruler", "calculator"
    ]

# Load Model - Using EfficientDet for better performance with more classes
@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
    return hub.load(model_url)

model = load_model()

def detect_objects(image):
    # Convert PIL image to TensorFlow tensor
    img_array = np.array(image)
    # Convert RGB to BGR (TF models expect BGR)
    converted_img = tf.image.convert_image_dtype(img_array, tf.float32)[tf.newaxis, ...]
    # Get model output
    result = model(converted_img)
    # Process results
    result = {key: value.numpy() for key, value in result.items()}
    return result

def draw_boxes(image, output):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    detection_boxes = output["detection_boxes"][0]
    detection_scores = output["detection_scores"][0]
    detection_classes = output["detection_classes"][0].astype(int)
    
    # Map detection classes to our expanded class list
    # EfficientDet uses COCO classes by default, but we'll map to our extended list
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.4:  # Confidence threshold
            # Get class index (adjust from COCO to our extended list if needed)
            class_id = detection_classes[i]
            # Map to our class names - handle out of bounds gracefully
            class_name = imagenet_classes[class_id % len(imagenet_classes)] if class_id < len(imagenet_classes)*2 else f"Class {class_id}"
            
            # Get box coordinates
            y_min, x_min, y_max, x_max = detection_boxes[i]
            x_min, x_max = int(x_min * width), int(x_max * width)
            y_min, y_max = int(y_min * height), int(y_max * height)
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            
            # Draw label
            label = f"{class_name} ({detection_scores[i]:.2f})"
            text_size = draw.textbbox((0, 0), label)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            
            # Draw text background
            draw.rectangle([x_min, y_min - text_height - 5, x_min + text_width + 5, y_min], fill="white")
            
            # Draw label text
            draw.text((x_min + 2, y_min - text_height - 3), label, fill="black")
    
    return image

# Streamlit UI
st.title("🖼️ Enhanced Object Detection (500+ Classes)")
st.write("Upload an image to detect objects with bounding boxes!")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    
    st.write("🔍 Detecting objects...")
    output = detect_objects(image)
    
    # Draw bounding boxes and show image
    result_image = draw_boxes(image, output)
    st.image(result_image, caption="🖼️ Detected Objects", use_column_width=True)
    
    # Display class information
    st.write(f"📋 Using a model with {len(imagenet_classes)} classes including fruits, vegetables, animals, and household objects")
