import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageFile
import textwrap

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_images(dir_name):

    # Get a list of all image file paths in the folder
    image_paths = [os.path.join(dir_name, file) for file in os.listdir(dir_name) if file.endswith(('.jpg', '.png', 'jpeg'))]

    # Calculate the number of rows and columns needed for the subplots
    num_images = len(image_paths)
    num_cols = 5  # Set the desired number of columns
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate the required number of rows

    # Create a figure and axis objects for subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axes = axes.ravel()  # Flatten the axes object to make indexing easier

    # Loop through the images and display them in the subplots
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')

    # Remove any empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    # Adjust subplot spacing and display the figure
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def truncate_text(text, max_width, font_size, dpi):
    """Truncate text to fit within a given width."""
    fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi)
    t = ax.text(0, 0, text, fontsize=font_size)
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    plt.close(fig)

    if bbox.width > max_width:
        ratio = max_width / bbox.width
        max_chars = int(len(text) * ratio)
        return text[:max_chars-3] + '...'
    return text

def plot_results(df):
    plt.style.use('seaborn-v0_8')
    # Calculate the number of rows and columns for the subplot grid
    n_images = len(df)
    n_cols = min(5, n_images)  # Max 5 columns
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 12*n_rows), dpi=100)
    fig.tight_layout(pad=1.0)

    # Flatten the axes array if it's 2D
    if n_rows > 1:
        axes = axes.flatten()

    # Iterate through the DataFrame and plot each image
    for i, (_, row) in enumerate(df.iterrows()):
        if i < len(axes):
            # Read the image
            img = Image.open(row['image_path'])
            
            # Plot the image
            if n_rows == 1:
                ax = axes[i] if n_cols > 1 else axes
            else:
                ax = axes[i]
            
            ax.imshow(img)
            ax.axis('off')
            
            # Get the width of the image in pixels
            img_width = ax.get_window_extent().width
            
            # Truncate and wrap the text
            truncated_text = truncate_text(row['text'], img_width, 8, fig.dpi)
            wrapped_text = textwrap.fill(truncated_text, width=40)
            
            # Set the title (text)
            ax.set_title(wrapped_text, fontsize=18, wrap=True, y=1.05)

    # Remove any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()

    
def plot_scatter_plot(title, full_data, new_data):
    # Set the style to a dark background
    plt.style.use('dark_background')

    # Create a new figure with a specific background color
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1C1C1C')
    ax.set_facecolor('#1C1C1C')
    
    colors = ['#FF6B6B', '#4ECDC4', '#98FB98', '#FFD700', '#FF69B4', '#90EE90', '#E6E6FA', '#ADD8E6', '#FF00FF', '#FFA07A']
    colors_index = 0
    
    for key in full_data:
        ax.scatter(full_data[key][:, 0], full_data[key][:, 1], c=colors[colors_index], label=key, s=75, edgecolors='white')
        colors_index +=1
    
    for key in new_data:
        ax.scatter(new_data[key][:, 0], new_data[key][:, 1], c=colors[colors_index], label=key, s=200, edgecolors='white')
        colors_index +=1
    
#     ax.scatter(reduced_cars_embeddings[:, 0], reduced_cars_embeddings[:, 1], c='#4ECDC4', label='Cars', s=75, edgecolors='white')
    
    
#     ax.scatter(reduced_new_car_embeddings[:, 0], reduced_new_car_embeddings[:, 1], c='#FFA07A', label='New Car Image', s=150, edgecolors='white')
#     ax.scatter(reduced_new_cat_embeddings[:, 0], reduced_new_cat_embeddings[:, 1], c='#98FB98', label='New Cat Image', s=150, edgecolors='white')
#     ax.scatter(reduced_car_text_embeddings[:, 0], reduced_car_text_embeddings[:, 1], c='#FFD700', label='New Car Text', s=150, edgecolors='white')
#     ax.scatter(reduced_cat_text_embeddings[:, 0], reduced_cat_text_embeddings[:, 1], c='#FF69B4', label='New Cat Text', s=150, edgecolors='white')

    # Customize labels and title
    ax.set_xlabel('Feature 1', fontsize=18, color='white')
    ax.set_ylabel('Feature 2', fontsize=18, color='white')
    ax.set_title(title, fontsize=20, color='white', fontweight='bold')

    # Customize legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor='#2F2F2F', edgecolor='none', fontsize=14)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.3)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    
    
def reduce_dimensionality(array, n_components=2):
    from sklearn.decomposition import PCA

    # Create a PCA object with 2 components
    pca = PCA(n_components=n_components)

    # Fit the PCA model to your data and transform it
    return pca.fit(array)


def process_images(folder_path):
    # Set the maximum file size (in bytes) and resolution
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_RESOLUTION = (720, 720)

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.png', 'jpeg')):
            # Construct the full path to the image file
            image_path = os.path.join(folder_path, filename)

            # Open the image
            with Image.open(image_path) as img:
                # Check if the image exceeds the maximum file size
                if os.path.getsize(image_path) > MAX_FILE_SIZE:
                    # Resize the image while preserving the aspect ratio
                    img.thumbnail(MAX_RESOLUTION)

                    # Save the resized image
                    img.save(image_path)
                    print(f"Resized {filename} due to file size.")

                # Check if the image exceeds the maximum resolution
                elif img.size > MAX_RESOLUTION:
                    # Resize the image while preserving the aspect ratio
                    img.thumbnail(MAX_RESOLUTION)

                    # Save the resized image
                    img.save(image_path)
                    print(f"Resized {filename} due to resolution.")

                else:
                    print(f"{filename} is already compliant.")
    
def pdf2imgs(pdf_path, pdf_pages_dir="content/Accessibility/pdf_pages"):
    """
    Convert a PDF file to individual PNG images for each page.

    Args:
        pdf_path (str): The path to the PDF file.
        pdf_pages_dir (str, optional): The directory to save the PNG images. Defaults to "content/Accessibility/pdf_pages".

    Returns:
        str: The path to the directory containing the PNG images.
    """
    import pypdfium2 as pdfium
    # Open the PDF document
    pdf = pdfium.PdfDocument(pdf_path)

    # Create the directory to save the PNG images if it doesn't exist
    os.makedirs(pdf_pages_dir, exist_ok=True)

    # Get the resolution of the first page to determine the scale factor
    resolution = pdf.get_page(0).render().to_numpy().shape
    scale = 1 if max(resolution) >= 1620 else 300 / 72  # Scale factor based on resolution

    # Get the number of pages in the PDF
    n_pages = len(pdf)

    # Loop through each page and save as a PNG image
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        pil_image = page.render(
            scale=scale,
            rotation=0,
            crop=(0, 0, 0, 0),
            may_draw_forms=False,
            fill_color=(255, 255, 255, 255),
            draw_annots=False,
            grayscale=False,
        ).to_pil()
        image_path = os.path.join(pdf_pages_dir, f"page_{page_number:03d}.png")
        pil_image.save(image_path)

    return pdf_pages_dir
                   