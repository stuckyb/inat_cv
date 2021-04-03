# Copyright (C) 2020 Brian J. Stucky
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import unittest
from torchvision import transforms
from data import ImageCsvDataset


class TestImageCsvDataset(unittest.TestCase):
    def test_init(self):
        ds = ImageCsvDataset(
            'images/labels.csv', 'images/12_images', 'fname', 'is_mono'
        )

        self.assertEqual(12, len(ds))
        self.assertEqual('black.png', ds.x[0])
        self.assertEqual(['N', 'Y'], ds.classes)
        self.assertEqual([1,0,0,0,0,1,0,0,0,0,0,1], ds.y)

    def test_getitem(self):
        ds = ImageCsvDataset(
            'images/labels.csv', 'images/12_images', 'fname', 'is_mono',
            transform=None
        )

        self.assertEqual((448, 448), ds[0][0].size)

        ds = ImageCsvDataset(
            'images/labels.csv', 'images/12_images', 'fname', 'is_mono',
            transform=transforms.Compose([
                transforms.Resize((224, 224))
            ])
        )

        self.assertEqual((224, 224), ds[0][0].size)

