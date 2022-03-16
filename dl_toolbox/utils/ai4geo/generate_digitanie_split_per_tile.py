import csv
import imagesize
import os

with open('split_scenario_2.csv', 'w', newline='') as csvfile:
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
            'Paris',
            'Montpellier'
    ]:
        #if city == 'Toulouse':
        #    for i, tile in enumerate([
        #        'empalot', 
        #        'arenes', 
        #        'bagatelle', 
        #        'cepiere', 
        #        'lardenne', 
        #        'minimes', 
        #        'mirail',
        #        'montaudran', 
        #        'zenith', 
        #        'ramier'
        #    ]):
        #        writer.writerow(
        #            [
        #                'Toulouse',
        #                i+1,
        #                os.path.join(path, city, f'tlse_{tile}_img_c.tif'),
        #                os.path.join(path, city, f'tlse_{tile}_c.tif'),
        #                0,
        #                0,
        #                2000,
        #                2000,
        #                (i+1) % 5
        #            ]
        #        )
        #else:
        for i in range(1, 11):
            img_path = os.path.join(path, city, city.lower()+f'_tuile_{i}_img_normalized.tif')
            label_path = os.path.join(path, city, city.lower()+f'_tuile_{i}.tif')
            width, height = imagesize.get(img_path)
            writer.writerow(
                [
                    city,
                    i,
                    img_path,
                    label_path,
                    0,
                    0,
                    width,
                    height,
                    i % 5
                ]
            )
