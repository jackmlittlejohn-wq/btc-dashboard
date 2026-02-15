#!/usr/bin/env python3
"""Generate simple app icons for PWA"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create eccentric JL icon with stylized gold text on black background"""
    # Create image with black background
    img = Image.new('RGB', (size, size), color='#000000')
    draw = ImageDraw.Draw(img)

    # Try to use bold italic font for flair
    try:
        font_size = int(size * 0.55)  # Larger text
        try:
            # Try Arial Black or Bold Italic first for more flair
            font = ImageFont.truetype("BRADHITC.TTF", font_size)  # Bradley Hand
        except:
            try:
                font = ImageFont.truetype("arialbd.ttf", font_size)  # Arial Bold
            except:
                try:
                    font = ImageFont.truetype("Arial-BoldMT.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                    except:
                        font = ImageFont.truetype("arial.ttf", font_size)

        # Text to draw
        text = "JL"

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - bbox[1]

        # Multi-layer shadow for dramatic effect
        shadow_layers = [
            (8, '#1a0f00'),  # Deepest shadow (dark brown)
            (6, '#4d2600'),  # Mid shadow (brown)
            (4, '#805000'),  # Light shadow (dark gold)
            (2, '#b38600'),  # Glow layer (medium gold)
        ]

        for offset, color in shadow_layers:
            offset_scaled = (size // 60) * offset
            draw.text((x + offset_scaled, y + offset_scaled), text, fill=color, font=font)

        # Main gold gradient effect (simulate with multiple colors)
        # Highlight (bright gold)
        draw.text((x - 1, y - 1), text, fill='#FFED4E', font=font)  # Light gold highlight
        # Main text (rich gold)
        draw.text((x, y), text, fill='#FFD700', font=font)

        # Add sparkle effect with small accents
        accent_offset = size // 6
        accent_size = size // 30
        # Top right sparkle
        draw.ellipse([x + text_width - accent_offset, y,
                     x + text_width - accent_offset + accent_size, y + accent_size],
                    fill='#FFFFFF')
        # Bottom left sparkle
        draw.ellipse([x + accent_offset//2, y + text_height - accent_size,
                     x + accent_offset//2 + accent_size, y + text_height],
                    fill='#FFED4E')

    except Exception as e:
        print(f"Could not add text: {e}")
        # Fallback with basic styling
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
