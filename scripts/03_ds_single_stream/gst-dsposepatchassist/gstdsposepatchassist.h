/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#ifndef __GST_DSPOSEPATCHASSIST_H__
#define __GST_DSPOSEPATCHASSIST_H__

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

#define PACKAGE "dsposepatchassist"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "DeepStream pose-patch proxy assist plugin"
#define BINARY_PACKAGE "AID DeepStream pose-patch proxy assist plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstDsPosePatchAssist GstDsPosePatchAssist;
typedef struct _GstDsPosePatchAssistClass GstDsPosePatchAssistClass;

struct PosePatchGeometry
{
  cv::Rect2f patch_rect_proc;
  cv::Point2f anchor_proc;
  std::string source;
};

struct PosePatchTrackState
{
  guint source_id;
  guint64 track_id;
  guint64 last_real_frame_num;
  guint proxy_age_frames;
  guint proxy_fail_count;
  bool proxy_active;
  cv::Rect2f last_real_bbox_proc;
  cv::Rect2f last_bbox_proc;
  cv::Rect2f last_patch_rect_proc;
  std::string patch_source;
  std::vector<cv::Point2f> patch_points;
};

#define GST_TYPE_DSPOSEPATCHASSIST (gst_dsposepatchassist_get_type())
#define GST_DSPOSEPATCHASSIST(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSPOSEPATCHASSIST,GstDsPosePatchAssist))
#define GST_DSPOSEPATCHASSIST_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSPOSEPATCHASSIST,GstDsPosePatchAssistClass))
#define GST_DSPOSEPATCHASSIST_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSPOSEPATCHASSIST, GstDsPosePatchAssistClass))
#define GST_IS_DSPOSEPATCHASSIST(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSPOSEPATCHASSIST))
#define GST_IS_DSPOSEPATCHASSIST_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSPOSEPATCHASSIST))

#define NVDSEXAMPLE_MAX_BATCH_SIZE 1024

struct _GstDsPosePatchAssist
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
  gdouble patch_width_ratio;
  gdouble patch_height_ratio;
  gdouble patch_y_offset_ratio;
  guint proxy_fail_tolerance;
  guint min_patch_size_px;
  gint person_class_id;

  std::unordered_map<std::string, PosePatchTrackState> *track_states;
  cv::Mat *prev_gray;
  gchar *sidecar_path;
  std::ofstream *sidecar_stream;
};

struct _GstDsPosePatchAssistClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dsposepatchassist_get_type (void);

G_END_DECLS

#endif  /* __GST_DSPOSEPATCHASSIST_H__ */
