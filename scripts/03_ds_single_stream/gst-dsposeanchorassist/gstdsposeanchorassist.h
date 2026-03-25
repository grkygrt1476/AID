/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#ifndef __GST_DSPOSEANCHORASSIST_H__
#define __GST_DSPOSEANCHORASSIST_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "gst-nvquery.h"
#include "gstnvdsinfer.h"
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "nvdsinfer.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#define PACKAGE "dsposeanchorassist"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "DeepStream pose-anchor assist plugin"
#define BINARY_PACKAGE "AID DeepStream pose-anchor assist plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstDsPoseAnchorAssist GstDsPoseAnchorAssist;
typedef struct _GstDsPoseAnchorAssistClass GstDsPoseAnchorAssistClass;

struct PoseAnchorPatchGeometry
{
  cv::Rect2f patch_rect_proc;
  cv::Point2f anchor_proc;
  std::string patch_source;
  std::string pose_anchor_source;
};

struct PoseAnchorTrackState
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
  cv::Point2f last_anchor_proc;
  cv::Point2f last_flow_delta_proc;
  float last_flow_mag_proc;
  std::string patch_source;
  std::string pose_anchor_source;
  std::string last_failure_reason;
  std::vector<cv::Point2f> patch_points;
};

#define GST_TYPE_DSPOSEANCHORASSIST (gst_dsposeanchorassist_get_type())
#define GST_DSPOSEANCHORASSIST(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSPOSEANCHORASSIST,GstDsPoseAnchorAssist))
#define GST_DSPOSEANCHORASSIST_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSPOSEANCHORASSIST,GstDsPoseAnchorAssistClass))
#define GST_DSPOSEANCHORASSIST_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSPOSEANCHORASSIST, GstDsPoseAnchorAssistClass))
#define GST_IS_DSPOSEANCHORASSIST(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSPOSEANCHORASSIST))
#define GST_IS_DSPOSEANCHORASSIST_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSPOSEANCHORASSIST))

#define NVDSEXAMPLE_MAX_BATCH_SIZE 1024

struct _GstDsPoseAnchorAssist
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
  guint pose_sgie_uid;
  gdouble pose_min_keypoint_conf;

  std::unordered_map<std::string, PoseAnchorTrackState> *track_states;
  cv::Mat *prev_gray;
  gchar *sidecar_path;
  std::ofstream *sidecar_stream;
};

struct _GstDsPoseAnchorAssistClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dsposeanchorassist_get_type (void);

G_END_DECLS

#endif  /* __GST_DSPOSEANCHORASSIST_H__ */
