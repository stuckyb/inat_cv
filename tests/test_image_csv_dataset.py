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
from data import ImageCsvDataset, getDatasets, getDataLoaders


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

    def test_getBalancedSampleWeights(self):
        ds = ImageCsvDataset(
            'images/labels.csv', 'images/12_images', 'fname', 'is_mono'
        )

        weights = ds.getBalancedSampleWeights()
        exp_weights = [12/3] + [12/9]*4 + [12/3] + [12/9]*5 + [12/3]
        self.assertEqual(12, len(weights))
        self.assertAlmostEqual(len(ds) * 2, sum(weights))
        self.assertEqual(exp_weights, weights)

        # Labels at these indices should be 0,1,0,0,1,0.
        weights = ds.getBalancedSampleWeights([4,5,8,10,11,1])
        exp_weights = [1.5, 3, 1.5, 1.5, 3, 1.5]
        self.assertEqual(6, len(weights))
        self.assertAlmostEqual(len(ds), sum(weights))
        self.assertEqual(exp_weights, weights)

        # Labels at these indices should be 0,0,0,0.
        weights = ds.getBalancedSampleWeights([4,8,10,1])
        exp_weights = [1, 1, 1, 1]
        self.assertEqual(4, len(weights))
        self.assertAlmostEqual(4, sum(weights))
        self.assertEqual(exp_weights, weights)


class TestGetDatasets(unittest.TestCase):
    def test_getDatasets(self):
        train_ds, val_ds = getDatasets(
            'images/labels.csv', 'images/12_images', 'fname', 'is_mono'
        )

        self.assertEqual(9, len(train_ds))
        self.assertEqual(3, len(val_ds))

        train_ds, val_ds = getDatasets(
            'images/labels.csv', 'images/12_images', 'fname', 'is_mono',
            train_size=1.0
        )

        self.assertEqual(12, len(train_ds))
        self.assertEqual(0, len(val_ds))

    def test_getDataLoaders(self):
        train_ds, val_ds = getDatasets(
            'images/labels.csv', 'images/12_images', 'fname', 'is_mono',
            train_size=1.0, train_transform=transforms.ToTensor()
        )

        train_dl, val_dl = getDataLoaders(
            train_ds, val_ds, batch_size=1, num_workers=2,
            use_weighted_sampling=True
        )

        # Verify that weighted sampling is working.
        counts = [0, 0]

        for i in range(100):
            for samp in train_dl:
                label = int(samp[1][0])
                counts[label] += 1

        counts[0] /= 100 * 12
        counts[1] /= 100 * 12
        print(counts)

        self.assertLess(abs(0.5 - counts[0]), 0.03)

