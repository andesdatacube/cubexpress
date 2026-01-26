"""Tests for animation module."""

import pytest

from cubexpress.processing.animation import create_gif


class TestCreateGif:
    """Tests for create_gif function."""

    @pytest.fixture
    def sample_images(self, tmp_path):
        """Create sample PNG images for testing."""
        from PIL import Image

        paths = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        for i, color in enumerate(colors):
            img = Image.new("RGB", (100, 100), color)
            path = tmp_path / f"frame_{i}.png"
            img.save(path)
            paths.append(path)

        return paths

    def test_creates_gif(self, sample_images, tmp_path):
        output = tmp_path / "test.gif"
        result = create_gif(sample_images, output)

        assert result == output
        assert output.exists()

    def test_gif_has_correct_frames(self, sample_images, tmp_path):
        from PIL import Image

        output = tmp_path / "test.gif"
        create_gif(sample_images, output)

        with Image.open(output) as gif:
            assert gif.n_frames == 3

    def test_empty_paths_raises(self, tmp_path):
        output = tmp_path / "test.gif"
        with pytest.raises(ValueError, match="No images"):
            create_gif([], output)

    def test_handles_rgba_images(self, tmp_path):
        from PIL import Image

        # Create RGBA image with transparency
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        path = tmp_path / "rgba.png"
        img.save(path)

        output = tmp_path / "test.gif"
        result = create_gif([path], output)

        assert result.exists()

    def test_custom_duration(self, sample_images, tmp_path):
        output = tmp_path / "test.gif"
        create_gif(sample_images, output, duration=1000)

        assert output.exists()

    def test_custom_background_color(self, tmp_path):
        from PIL import Image

        # Create RGBA with transparency
        img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        path = tmp_path / "transparent.png"
        img.save(path)

        output = tmp_path / "test.gif"
        create_gif([path], output, background_color=(255, 255, 255))

        assert output.exists()
