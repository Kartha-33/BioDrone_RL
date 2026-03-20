import unittest
import numpy as np
import sys
import os

# Path hack to import src
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from models.vision import FlyRetina

class TestVision(unittest.TestCase):
    def setUp(self):
        # Create a standard retina: 32 pixels, 120-degree FOV
        self.retina = FlyRetina(num_photoreceptors=32, fov_deg=120)

    def test_center_view_is_symmetric(self):
        """
        If the drone is exactly in the middle (x=0), the left view should mirror the right view.
        """
        # Render from center
        view = self.retina.render(lateral_pos=0.0)
        
        # Split into left and right halves
        mid = len(view) // 2
        left_half = view[:mid]
        right_half = view[mid:]
        
        # The right half should mirror the reversed left half
        # (Note: exact pixel match depends on even/odd and ray angles, but structurally checks out)
        # Using a loose tolerance because of angle discretization
        np.testing.assert_allclose(left_half, right_half[::-1], atol=0.1)

    def test_left_wall_intensity(self):
        """
        If we move closer to the LEFT wall (x -> -1.0), the LEFT pixels (indices 0..N/2) 
        should be brighter than the RIGHT pixels.
        """
        # Move drone to x = -0.8 (close to left wall at -1.0)
        view = self.retina.render(lateral_pos=-0.8)
        
        mid = len(view) // 2
        avg_left_brightness = np.mean(view[:mid])
        avg_right_brightness = np.mean(view[mid:])
        
        print(f"\n[Vision Debug] Left Brightness: {avg_left_brightness:.4f}, Right: {avg_right_brightness:.4f}")
        
        self.assertGreater(avg_left_brightness, avg_right_brightness, 
                           "Left side should be brighter when close to left wall.")

    def test_crash_intensity(self):
        """
        If we are touching the wall, intensity should be max (1.0).
        """
        # Touching left wall
        view = self.retina.render(lateral_pos=-1.0)
        
        # The extreme left ray (index 0) should see the wall immediately
        self.assertAlmostEqual(view[0], 1.0, delta=0.2)

if __name__ == '__main__':
    unittest.main()