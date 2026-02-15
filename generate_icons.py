#!/usr/bin/env python3
"""Generate simple app icons for PWA"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple Bitcoin icon"""
    # Create image with black background
    img = Image.new('RGB', (size, size), color='#000000')
    draw = ImageDraw.Draw(img)

    # Draw orange circle
    margin = size // 8
    circle_bbox = [margin, margin, size - margin, size - margin]
    draw.ellipse(circle_bbox, fill='#f7931a', outline='#f7931a')

    # Draw Bitcoin symbol (₿)
    try:
        # Try to use a system font
        font_size = size // 2
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()

        # Draw ₿ symbol
        text = "₿"
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - bbox[1]

        draw.text((x, y), text, fill='#000000', font=font)
    except Exception as e:
        print(f"Could not add text: {e}")
        # Draw simple B if font fails
        draw.text((size//3, size//3), "B", fill='#000000')

    # Save image
    img.save(output_path, 'PNG')
    print(f"Created {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(script_dir, 'static')

    # Create icons
    create_icon(192, os.path.join(static_dir, 'icon-192.png'))
    create_icon(512, os.path.join(static_dir, 'icon-512.png'))

    print("Icons generated successfully!")
