/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#ifndef __GST_DSKLTASSIST_H__
#define __GST_DSKLTASSIST_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "gst-nvquery.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#define PACKAGE "dskltassist"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "DeepStream KLT proxy assist plugin"
#define BINARY_PACKAGE "AID DeepStream KLT proxy assist plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstDsKltAssist GstDsKltAssist;
typedef struct _GstDsKltAssistClass GstDsKltAssistClass;

struct KltTrackState
{
  guint source_id;
  guint64 track_id;
  guint64 last_real_frame_num;
  guint proxy_age_frames;
  bool proxy_active;
  cv::Rect2f last_real_bbox_proc;
  cv::Rect2f last_bbox_proc;
  std::vector<cv::Point2f> features;
};

#define GST_TYPE_DSKLTASSIST (gst_dskltassist_get_type())
#define GST_DSKLTASSIST(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSKLTASSIST,GstDsKltAssist))
#define GST_DSKLTASSIST_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSKLTASSIST,GstDsKltAssistClass))
#define GST_DSKLTASSIST_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSKLTASSIST, GstDsKltAssistClass))
#define GST_IS_DSKLTASSIST(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSKLTASSIST))
#define GST_IS_DSKLTASSIST_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSKLTASSIST))

#define NVDSEXAMPLE_MAX_BATCH_SIZE 1024

struct _GstDsKltAssist
{
  GstBaseTransform base_trans;

  GstVideoInfo video_info;
  NvBufSurface *inter_buf;
  NvBufSurfTransformConfigParams transform_config_params;
  cudaStream_t cuda_stream;
  guint is_integrated;
  guint unique_id;
  gint processing_width;
  gint processing_height;
  guint gpu_id;
  guint max_batch_size;
  gboolean process_full_frame;
  gboolean blur_objects;

  guint proxy_ttl_frames;
  gdouble max_center_shift_px;
  guint min_good_points;
  guint feature_max_corners;
  guint lk_win_size;
  gint person_class_id;

  std::unordered_map<std::string, KltTrackState> *track_states;
  cv::Mat *prev_gray;
  gchar *sidecar_path;
  std::ofstream *sidecar_stream;
};

struct _GstDsKltAssistClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dskltassist_get_type (void);

G_END_DECLS

#endif  /* __GST_DSKLTASSIST_H__ */
