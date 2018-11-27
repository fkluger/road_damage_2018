# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.datasets.imdb import imdb
import lib.datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid
from .rddc_eval import rddc_eval, rddc_eval_confusion
from lib.model.config import cfg


class road_damage(imdb):
    def __init__(self, image_set, year):
        name = 'rddc_' + year + '_' + image_set
        imdb.__init__(self, name)
        self._rgba = True if 'rgba' in year else False
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path()
        self._data_path = self._devkit_path
        self._classes = ('__background__',  # always index 0
                         'd00', 'd01', 'd10', 'd11', 'd20', 'd40', 'd43', 'd44')
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self._image_ext = '.jpg' if not self._rgba else ".png"
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'


        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'matlab_eval': False,
                       'rpn_file': None}

        assert os.path.exists(self._devkit_path), \
            'devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'JPEGImages' if not self._rgba else 'JPEGImagesRGBA',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'rddc_' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                try:
                    roidb = pickle.load(fid)
                except:
                    roidb = pickle.load(fid, encoding='bytes')
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        gt_roidb = [x for x in gt_roidb if x is not None]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        num_objs = 0
        for ix, obj in enumerate(objs):
            class_name = obj.find('name').text.lower().strip()
            if class_name == 'd30':
                continue
            num_objs += 1

        # if num_objs == 0:
        #     return None

        assert num_objs > 0, "%s" % filename

        # num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        print("set: ", self._image_set)
        ix = 0
        for ix2, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            print(index)
            class_name = obj.find('name').text.lower().strip()
            if class_name == 'd30':
                continue
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
            ix += 1

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
                   else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _get_rddc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_rddc_result.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} rddc results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _write_rddc_results_file(self, all_boxes):

        filename = self._get_rddc_results_file_template()
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                all_dets = []
                for cls_ind, cls in enumerate(self.classes):
                    all_dets.append((all_boxes[cls_ind][im_ind], cls_ind))
                if all_dets == []:
                    continue

                good_dets = []
                good_classes = []
                for dets_and_cls in all_dets:
                    dets = dets_and_cls[0]
                    cls_ind = dets_and_cls[1]
                    for det in dets:
                        if det[4] >= 0.95:
                            good_dets.append(det)
                            good_classes.append(cls_ind)

                if len(good_dets) > 0:
                    good_dets = np.vstack(good_dets)
                    good_classes = np.vstack(good_classes)
                    sorting = np.argsort(good_dets[:, 4])[::-1]
                    good_dets = good_dets[sorting, :]
                    good_classes = good_classes[sorting, :]
                    if good_dets.shape[0] > 5:
                        good_dets = good_dets[0:5, :]

                    write_string = "%s.jpg," % index

                    for k in range(good_dets.shape[0]):
                        if good_classes[k] < 9:
                            write_string += "%d %d %d %d %d " % (good_classes[k], good_dets[k, 0] + 1, good_dets[k, 1] + 1,
                                                             good_dets[k, 2] + 1, good_dets[k, 3] + 1)

                    write_string += "\n"
                    f.write(write_string)
                else:
                    write_string = "%s.jpg,\n" % index
                    f.write(write_string)

        print("dets: ", len(dets))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        true_pos = 0
        false_pos = 0
        num_pos = 0

        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            elif cls in []:
                ignore = True
            else:
                ignore = False

            filename = self._get_voc_results_file_template()#.format(cls)
            # print("detpath: ", filename)
            rec, prec, ap, tp, fp, npos, tp1, fp1 = rddc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=False, use_diff=True, ignore=ignore)
            aps += [ap]
            true_pos += tp1
            false_pos += fp1
            num_pos += npos
            if tp1+fp1 > 0:
                prec = tp1 / (tp1 + fp1)
            else:
                prec = 0
            rec = tp1 / npos
            f1 = 2 * rec * prec / (rec + prec)
            print(('F1 for %s = %.4f' % (cls, f1)))
            print(('AP for {} = {:.4f}'.format(cls, ap)))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        cachedir = os.path.join(self._devkit_path, 'annotations_cache_confusion')
        confusion, confused_items = rddc_eval_confusion(self._get_voc_results_file_template(), annopath, imagesetfile, self._classes, cachedir)
        print(confusion)

        first_line = "---- "
        for cls in self._classes:
            if cls == "__background__":
                cls = "bg"
            first_line += "{:>4.4s} ".format(cls)
        # print(first_line)
        for k in range(confusion.shape[0]):
            row = confusion[k,:].squeeze()
            line = "{:4.4s} ".format(self._classes[k] if self._classes[k] != "__background__" else "bg")
            for l in range(row.shape[0]):
                line += "{:4d} ".format(row[l])
            # print(line)

        for cls_det, confused_row in zip(self._classes, confused_items):
            cls_det = "bkg" if cls_det == '__background__' else cls_det
            for cls_tru, confused_cell in zip(self._classes, confused_row):
                cls_tru = "bkg" if cls_tru == '__background__' else cls_tru
                confused_path = "/home/kluger/tmp/confused_{}_as_{}.txt".format(cls_tru, cls_det)
                with open(confused_path, "w") as text_file:
                    for confused_item in confused_cell:
                        # print(confused_item)
                        print("{} {:d} {:d} {:d} {:d}".format(confused_item[0], confused_item[1][0], confused_item[1][1],
                                                              confused_item[1][2], confused_item[1][3]), file=text_file)

        prec = true_pos / (true_pos + false_pos)
        rec = true_pos / num_pos
        F1 = 2*rec*prec/(rec+prec)

        print(('Mean AP = {:.4f}'.format(np.mean(aps))))
        print(('Tot. F1 = %.4f' % (F1)))
        print(('true_pos: %d' % (true_pos)))
        print(('false_pos: %d' % (false_pos)))
        print(('num_pos: %d' % (num_pos)))
        # exit(0)
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print(('{:.3f}'.format(ap)))
        print(('{:.3f}'.format(np.mean(aps))))
        print('~~~~~~~~')
        print('')
        # print('--------------------------------------------------------------')
        # print('Results computed with the **unofficial** Python eval code.')
        # print('Results should be very close to the official MATLAB eval code.')
        # print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        # print('-- Thanks, The Management')
        # print('--------------------------------------------------------------')
        return np.mean(aps), F1

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print(('Running:\n{}'.format(cmd)))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_rddc_results_file(all_boxes)
        self._write_voc_results_file(all_boxes)
        map, f1 = self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                # os.remove(filename)
        return map, f1

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    from datasets.pascal_voc import pascal_voc

    d = pascal_voc('trainval', '2007')
    res = d.roidb
    from IPython import embed;

    embed()
