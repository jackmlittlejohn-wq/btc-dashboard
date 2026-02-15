#!/usr/bin/env python3
"""Generate simple app icons for PWA"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create eccentric JL icon with stylized gold text on black background"""
    # Create image with black background
    img = Image.new('RGB', (size, size), color='#000000')
    draw = ImageDraw.Draw(img)

    # Try to use extra bold font for maximum impact
    try:
        font_size = int(size * 0.6)  # Even larger text
        try:
            # Try Impact or Arial Black first for maximum boldness
            font = ImageFont.truetype("impact.ttf", font_size)  # Impact - very bold
        except:
            try:
                font = ImageFont.truetype("Impact.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("ariblk.ttf", font_size)  # Arial Black
                except:
                    try:
                        font = ImageFont.truetype("arialbd.ttf", font_size)  # Arial Bold
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

        # Extra bold multi-layer shadow for dramatic 3D effect
        shadow_layers = [
            (10, '#0d0700'),  # Deepest shadow (almost black)
            (8, '#1a0f00'),   # Very deep shadow (dark brown)
            (6, '#331a00'),   # Deep shadow (brown)
            (5, '#4d2600'),   # Mid shadow (brown)
            (4, '#664400'),   # Light shadow (dark gold)
            (3, '#805000'),   # Lighter shadow (dark gold)
            (2, '#996600'),   # Glow layer (medium gold)
            (1, '#b38600'),   # Bright glow (medium-bright gold)
        ]

        for offset, color in shadow_layers:
            offset_scaled = (size // 50) * offset
            draw.text((x + offset_scaled, y + offset_scaled), text, fill=color, font=font)

        # Main gold gradient effect with multiple layers for richness
        # Deep gold base
        draw.text((x + 2, y + 2), text, fill='#CC9900', font=font)
        # Mid gold
        draw.text((x + 1, y + 1), text, fill='#E6AC00', font=font)
        # Bright highlight
        draw.text((x - 1, y - 1), text, fill='#FFED4E', font=font)
        # Main brilliant gold
        draw.text((x, y), text, fill='#FFD700', font=font)

        # Add multiple sparkle effects for extra flair
        accent_size = size // 25
        # Top right sparkle cluster
        sparkle_x = x + text_width - size // 8
        draw.ellipse([sparkle_x, y, sparkle_x + accent_size, y + accent_size], fill='#FFFFFF')
        draw.ellipse([sparkle_x + accent_size//2, y - accent_size//2,
                     sparkle_x + accent_size + accent_size//2, y + accent_size//2], fill='#FFED4E')
        # Bottom left sparkle
        draw.ellipse([x + size//10, y + text_height - accent_size,
                     x + size//10 + accent_size, y + text_height], fill='#FFFFFF')
        # Center sparkle
        draw.ellipse([x + text_width//2, y + text_height//3,
                     x + text_width//2 + accent_size//2, y + text_height//3 + accent_size//2],
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
