import csv
import os

with open('split_per_tile.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    path = '/work/OT/ai4geo/DATA/DATASETS/DIGITANIE'
    writer.writerow(['city',
                     'tile_id',
                     'img_path',
                     'label_path',
                     'x0',
                     'y0',
                     'patch_width',
                     'patch_height',
                     'fold_id'
                     ])
    for city in [
            'Toulouse',
            'Biarritz',
            'Strasbourg',
            'Paris'
    ]:
        if city == 'Toulouse':
            for i, tile in enumerate([
                'empalot', 
                'arenes', 
                'bagatelle', 
                'cepiere', 
                'lardenne', 
                'minimes', 
                'mirail',
                'montaudran', 
                'zenith', 
                'ramier'
            ]):
                writer.writerow(
                    [
                        'Toulouse',
                        i+1,
                        os.path.join(path, city, f'tlse_{tile}_img_c.tif'),
                        os.path.join(path, city, f'tlse_{tile}_c.tif'),
                        0,
                        0,
                        2000,
                        2000,
                        (i+1) % 5
                    ]
                )
        else:
            for i in range(1, 11):
                writer.writerow(
                    [
                        city,
                        i,
                        os.path.join(path, city, city.lower()+f'_tuile_{i}_img_c.tif'),
                        os.path.join(path, city, city.lower()+f'_tuile_{i}_c.tif'),
                        0,
                        0,
                        2000,
                        2000,
                        i % 5
                    ]
                )
