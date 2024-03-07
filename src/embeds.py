# Import the necessary libraries
import torch
from PIL import Image
import clip

# Load the model
# Use CUDA if available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained CLIP model and the associated preprocessing function
model, preprocess = clip.load("ViT-B/32", device=device)


# Define a preprocessing function to convert video frames into a format
# suitable for the model
def preprocess_frame(frame):
    # Convert the frame to a PIL Image and apply the preprocessing
    return preprocess(Image.fromarray(frame))


# Define a function to convert text into a format suitable for the model, and
# get its embedding
def embed_text(text):
    # Run the model in no_grad mode to prevent the computation graph from
    # being built (saves memory)
    with torch.no_grad():
        # Tokenize the text and move it to the appropriate device
        text_encoded = clip.tokenize([text]).to(device)

        # Pass the tokenized text through the model to get the text embeddings
        text_embedding = model.encode_text(text_encoded)
    return text_embedding


# Define a function to get the embeddings of a list of frames
def embed_frames(frames):
    # Initialize an empty list to store the frame embeddings
    frame_embeddings = []

    for frame in frames:
        # Run the model in no_grad mode to prevent the computation graph from
        # being built (saves memory)
        with torch.no_grad():
            # Preprocess the frame and add a batch dimension
            frame_preprocessed = preprocess_frame(frame).unsqueeze(0) \
                .to(device)

            # Pass the preprocessed frame through the model to get the frame
            # embeddings
            frame_embedding = model.encode_image(frame_preprocessed)

        # Add the frame embeddings to the list
        frame_embeddings.append(frame_embedding)

    # Stack the list of frame embeddings into a single tensor
    return torch.stack(frame_embeddings)
