#!/usr/bin/env python3
"""Generate simple app icons for PWA"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create minimalist JL icon with gold text on black background"""
    # Create image with black background
    img = Image.new('RGB', (size, size), color='#000000')
    draw = ImageDraw.Draw(img)

    # Try to use bold font for better effect
    try:
        font_size = int(size * 0.5)  # 50% of icon size
        try:
            # Try Arial Black or Bold first
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("Arial-Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()

        # Text to draw
        text = "JL"

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - bbox[1]

        # Draw gold text with subtle shadow for depth
        shadow_offset = size // 40  # Small shadow offset
        # Shadow (darker gold)
        draw.text((x + shadow_offset, y + shadow_offset), text, fill='#996515', font=font)
        # Main text (bright gold)
        draw.text((x, y), text, fill='#FFD700', font=font)

    except Exception as e:
        print(f"Could not add text: {e}")
        # Simple fallback
        draw.text((size//3, size//3), "JL", fill='#FFD700')

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
