#!/usr/bin/env python3
"""Generate anchor-style JL icons for PWA"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create symmetric JL icon that looks like an anchor with ocean gradient backdrop"""
    # Create image with ocean gradient background
    img = Image.new('RGB', (size, size))
    draw = ImageDraw.Draw(img)

    # Create deep ocean gradient background (dark blue to teal)
    for y in range(size):
        ratio = y / size
        # Deep ocean blue to lighter teal gradient
        r = int(15 + (ratio * 50))   # 15 -> 65
        g = int(30 + (ratio * 120))  # 30 -> 150
        b = int(80 + (ratio * 80))   # 80 -> 160
        draw.rectangle([0, y, size, y+1], fill=(r, g, b))

    # Add subtle wave/ripple effects
    wave_count = 5
    for i in range(wave_count):
        y_pos = int(size * (0.2 + i * 0.15))
        for x in range(size):
            wave_offset = int(5 * ((x / size) - 0.5) * ((x / size) - 0.5))
            alpha = int(20 - (abs(y_pos - (size * (0.2 + i * 0.15))) * 0.5))
            if alpha > 0:
                draw.point((x, y_pos + wave_offset), fill=(255, 255, 255))

    try:
        # Use extra bold font for anchor-like letters
        font_size = int(size * 0.65)
        try:
            font = ImageFont.truetype("impact.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("Impact.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("ariblk.ttf", font_size)
                except:
                    try:
                        font = ImageFont.truetype("arialbd.ttf", font_size)
                    except:
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                        except:
                            font = ImageFont.truetype("arial.ttf", font_size)

        # Text to draw - symmetric JL
        text = "JL"

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - bbox[1]

        # Create anchor effect with thick shadow (bottom heavy like an anchor)
        # Deep shadow layers - heavier at bottom
        shadow_layers = [
            (16, '#001a2e', 1.3),  # Deepest shadow with bottom weight
            (13, '#002640', 1.25), # Very deep shadow
            (10, '#003350', 1.2),  # Deep shadow
            (8, '#004060', 1.15),  # Mid-deep shadow
            (6, '#005580', 1.1),   # Mid shadow
            (4, '#006699', 1.05),  # Light shadow
        ]

        for offset, color, bottom_weight in shadow_layers:
            offset_scaled = (size // 50) * offset
            bottom_offset = int(offset_scaled * bottom_weight)  # Heavier bottom
            draw.text((x + offset_scaled // 2, y + bottom_offset), text, fill=color, font=font)

        # Metallic gold/brass effect for anchor metal
        # Base layers
        draw.text((x + 3, y + 3), text, fill='#8B6914', font=font)  # Dark brass
        draw.text((x + 2, y + 2), text, fill='#A0752D', font=font)  # Medium brass
        draw.text((x + 1, y + 1), text, fill='#C4963D', font=font)  # Light brass

        # Highlight on top left (where light would hit anchor metal)
        draw.text((x - 1, y - 1), text, fill='#FFD700', font=font)  # Gold highlight

        # Main metallic silver-gold color
        draw.text((x, y), text, fill='#E5C77F', font=font)  # Metallic brass-gold

        # Add anchor chain detail - small circle at top center of J
        chain_x = x + text_width // 4
        chain_y = y + size // 20
        chain_radius = size // 25

        # Chain link shadow
        draw.ellipse([chain_x + 2, chain_y + 2,
                     chain_x + chain_radius + 2, chain_y + chain_radius + 2],
                    fill='#003350')
        # Chain link
        draw.ellipse([chain_x, chain_y,
                     chain_x + chain_radius, chain_y + chain_radius],
                    fill='#C0C0C0', outline='#808080', width=max(1, size//200))

        # Add barnacle/marine details (small dots suggesting underwater anchor)
        barnacle_positions = [
            (x + text_width // 3, y + text_height * 2 // 3),
            (x + text_width * 2 // 3, y + text_height * 3 // 4),
            (x + text_width // 5, y + text_height // 2),
        ]
        for bx, by in barnacle_positions:
            barnacle_size = size // 40
            draw.ellipse([bx, by, bx + barnacle_size, by + barnacle_size],
                        fill='#4A5F6F', outline='#2A3F4F', width=1)

        # Add light rays/god rays from top
        ray_count = 8
        center_x = size // 2
        for i in range(ray_count):
            angle_offset = (i - ray_count // 2) * 0.3
            start_x = center_x + int(size * 0.2 * angle_offset)
            for j in range(3):
                ray_y = int(size * 0.15 * j)
                ray_width = int(size * 0.02)
                alpha_color = (255, 255, 255) if j == 0 else (200, 220, 240)
                draw.line([(start_x + j * angle_offset * 5, ray_y),
                          (center_x + j * angle_offset * 2, ray_y + size // 8)],
                         fill=alpha_color, width=max(1, ray_width - j))

    except Exception as e:
        print(f"Could not add text: {e}")
        # Fallback
        draw.text((size//4, size//3), "JL", fill='#E5C77F')

    # Save image
    img.save(output_path, 'PNG')
    print(f"Created {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(script_dir, 'static')

    # Create icons
    create_icon(192, os.path.join(static_dir, 'icon-192.png'))
    create_icon(512, os.path.join(static_dir, 'icon-512.png'))

    print("Anchor-style JL icons generated successfully!")
