/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#ifndef __GST_DSMONITOROSD_H__
#define __GST_DSMONITOROSD_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include "gstnvdsmeta.h"

#include <string>
#include <unordered_map>
#include <vector>

#define PACKAGE "dsmonitorosd"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "DeepStream monitoring OSD overlay plugin"
#define BINARY_PACKAGE "AID DeepStream monitoring OSD overlay plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstDsMonitorOsd GstDsMonitorOsd;
typedef struct _GstDsMonitorOsdClass GstDsMonitorOsdClass;

struct MonitorPoint
{
  gint x;
  gint y;
};

struct MonitorSourceConfig
{
  std::string label;
  std::string roi_status;
  std::vector<MonitorPoint> roi_points;
};

#define GST_TYPE_DSMONITOROSD (gst_dsmonitorosd_get_type())
#define GST_DSMONITOROSD(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSMONITOROSD,GstDsMonitorOsd))
#define GST_DSMONITOROSD_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSMONITOROSD,GstDsMonitorOsdClass))
#define GST_DSMONITOROSD_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSMONITOROSD, GstDsMonitorOsdClass))
#define GST_IS_DSMONITOROSD(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSMONITOROSD))
#define GST_IS_DSMONITOROSD_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSMONITOROSD))

#define NVDSEXAMPLE_MAX_BATCH_SIZE 1024

struct _GstDsMonitorOsd
{
  GstBaseTransform base_trans;

  GstVideoInfo video_info;
  guint unique_id;
  gint processing_width;
  gint processing_height;
  guint gpu_id;
  guint max_batch_size;
  gboolean process_full_frame;
  gboolean blur_objects;

  guint source_count;
  gint person_class_id;
  std::unordered_map<guint, MonitorSourceConfig> *source_configs;
};

struct _GstDsMonitorOsdClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dsmonitorosd_get_type (void);

G_END_DECLS

#endif  /* __GST_DSMONITOROSD_H__ */
