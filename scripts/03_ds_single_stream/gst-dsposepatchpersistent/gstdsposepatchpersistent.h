/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#ifndef __GST_DSPOSEPATCHPERSISTENT_H__
#define __GST_DSPOSEPATCHPERSISTENT_H__

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

#define PACKAGE "dsposepatchpersistent"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "DeepStream persistent pose-patch debug assist plugin"
#define BINARY_PACKAGE "AID DeepStream persistent pose-patch debug assist plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstDsPosePatchPersistent GstDsPosePatchPersistent;
typedef struct _GstDsPosePatchPersistentClass GstDsPosePatchPersistentClass;

struct PersistentPatchGeometry
{
  cv::Rect2f patch_rect_proc;
  cv::Point2f anchor_proc;
  std::string source;
};

struct PersistentTrackState
{
  guint source_id;
  guint64 track_id;
  guint64 last_real_frame_num;
  guint proxy_age_frames;
  guint last_tracked_points;
  guint frozen_hold_remaining_frames;
  bool proxy_active;
  bool frozen_hold_active;
  cv::Rect2f last_real_bbox_proc;
  cv::Rect2f last_bbox_proc;
  cv::Rect2f last_patch_rect_proc;
  cv::Point2f last_flow_delta_proc;
  float last_flow_mag_proc;
  std::string patch_source;
  std::string last_failure_reason;
  std::vector<cv::Point2f> patch_points;
};

#define GST_TYPE_DSPOSEPATCHPERSISTENT (gst_dsposepatchpersistent_get_type())
#define GST_DSPOSEPATCHPERSISTENT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSPOSEPATCHPERSISTENT,GstDsPosePatchPersistent))
#define GST_DSPOSEPATCHPERSISTENT_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSPOSEPATCHPERSISTENT,GstDsPosePatchPersistentClass))
#define GST_DSPOSEPATCHPERSISTENT_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSPOSEPATCHPERSISTENT, GstDsPosePatchPersistentClass))
#define GST_IS_DSPOSEPATCHPERSISTENT(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSPOSEPATCHPERSISTENT))
#define GST_IS_DSPOSEPATCHPERSISTENT_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSPOSEPATCHPERSISTENT))

#define NVDSEXAMPLE_MAX_BATCH_SIZE 1024

struct _GstDsPosePatchPersistent
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

  guint hard_max_proxy_age_frames;
  gdouble max_center_shift_px;
  guint min_good_points;
  guint feature_max_corners;
  guint lk_win_size;
  gdouble patch_width_ratio;
  gdouble patch_height_ratio;
  gdouble patch_y_offset_ratio;
  gboolean freeze_on_patch_fail;
  guint hold_after_fail_frames;
  guint min_patch_size_px;
  gint person_class_id;

  std::unordered_map<std::string, PersistentTrackState> *track_states;
  cv::Mat *prev_gray;
  gchar *sidecar_path;
  std::ofstream *sidecar_stream;
  std::vector<std::pair<gint, gint>> *roi_vertices_orig;
};

struct _GstDsPosePatchPersistentClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dsposepatchpersistent_get_type (void);

G_END_DECLS

#endif  /* __GST_DSPOSEPATCHPERSISTENT_H__ */
