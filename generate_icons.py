#!/usr/bin/env python3
"""Generate simple JL icons for PWA"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create simple JL in normal font on black background"""
    # Black background
    img = Image.new('RGB', (size, size), color='#000000')
    draw = ImageDraw.Draw(img)

    try:
        # Use normal Arial font
        font_size = int(size * 0.5)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    font = ImageFont.load_default()

        text = "JL"

        # Get text dimensions
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center position
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - bbox[1]

        # Just draw white text, nothing else
        draw.text((x, y), text, fill='#FFFFFF', font=font)

    except Exception as e:
        print(f"Could not create icon: {e}")
        # Fallback
        draw.text((size//4, size//3), "JL", fill='#FFFFFF')

    # Save image
    img.save(output_path, 'PNG')
    print(f"Created {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(script_dir, 'static')

    # Create icons
    create_icon(192, os.path.join(static_dir, 'icon-192.png'))
    create_icon(512, os.path.join(static_dir, 'icon-512.png'))

    print("Simple JL icons generated successfully!")
