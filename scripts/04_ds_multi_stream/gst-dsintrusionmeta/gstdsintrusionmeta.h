/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#ifndef __GST_DSINTRUSIONMETA_H__
#define __GST_DSINTRUSIONMETA_H__

#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>

#include "gstnvdsmeta.h"

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#define PACKAGE "dsintrusionmeta"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "DeepStream Stage 04.03 intrusion metadata export plugin"
#define BINARY_PACKAGE "AID DeepStream intrusion metadata export plugin"
#define URL "http://nvidia.com/"

G_BEGIN_DECLS

typedef struct _GstDsIntrusionMeta GstDsIntrusionMeta;
typedef struct _GstDsIntrusionMetaClass GstDsIntrusionMetaClass;

struct IntrusionPoint
{
  gint x;
  gint y;
};

struct IntrusionSourceConfig
{
  std::string label;
  std::string roi_status;
  std::vector<IntrusionPoint> roi_points;
};

#define GST_TYPE_DSINTRUSIONMETA (gst_dsintrusionmeta_get_type())
#define GST_DSINTRUSIONMETA(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DSINTRUSIONMETA,GstDsIntrusionMeta))
#define GST_DSINTRUSIONMETA_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DSINTRUSIONMETA,GstDsIntrusionMetaClass))
#define GST_DSINTRUSIONMETA_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DSINTRUSIONMETA, GstDsIntrusionMetaClass))
#define GST_IS_DSINTRUSIONMETA(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DSINTRUSIONMETA))
#define GST_IS_DSINTRUSIONMETA_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DSINTRUSIONMETA))

#define NVDSEXAMPLE_MAX_BATCH_SIZE 1024

struct _GstDsIntrusionMeta
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
  std::unordered_map<guint, IntrusionSourceConfig> *source_configs;

  gchar *sidecar_path;
  std::ofstream *sidecar_stream;
  guint64 sidecar_rows_written;
};

struct _GstDsIntrusionMetaClass
{
  GstBaseTransformClass parent_class;
};

GType gst_dsintrusionmeta_get_type (void);

G_END_DECLS

#endif  /* __GST_DSINTRUSIONMETA_H__ */
