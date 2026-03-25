/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 */

#include "gstdsintrusionmeta.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

GST_DEBUG_CATEGORY_STATIC (gst_dsintrusionmeta_debug);
#define GST_CAT_DEFAULT gst_dsintrusionmeta_debug

enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_PROCESS_FULL_FRAME,
  PROP_BATCH_SIZE,
  PROP_BLUR_OBJECTS,
  PROP_GPU_DEVICE_ID
};

#define DEFAULT_UNIQUE_ID 16
#define DEFAULT_PROCESSING_WIDTH 1920
#define DEFAULT_PROCESSING_HEIGHT 1080
#define DEFAULT_PROCESS_FULL_FRAME TRUE
#define DEFAULT_BLUR_OBJECTS FALSE
#define DEFAULT_GPU_ID 0
#define DEFAULT_BATCH_SIZE 4
#define DEFAULT_SOURCE_COUNT 4
#define DEFAULT_PERSON_CLASS_ID 0

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
static GstStaticPadTemplate gst_dsintrusionmeta_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

static GstStaticPadTemplate gst_dsintrusionmeta_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA, I420 }")));

#define gst_dsintrusionmeta_parent_class parent_class
G_DEFINE_TYPE (GstDsIntrusionMeta, gst_dsintrusionmeta, GST_TYPE_BASE_TRANSFORM);

static void gst_dsintrusionmeta_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_dsintrusionmeta_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean gst_dsintrusionmeta_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_dsintrusionmeta_start (GstBaseTransform * btrans);
static gboolean gst_dsintrusionmeta_stop (GstBaseTransform * btrans);
static GstFlowReturn gst_dsintrusionmeta_transform_ip (GstBaseTransform * btrans,
    GstBuffer * inbuf);

struct IntrusionFrameStats
{
  guint total_objects = 0;
  guint tracked_objects = 0;
  guint candidate_objects = 0;
  guint candidate_tracked_objects = 0;
};

static inline guint
read_env_uint (const char *name, guint default_value, guint min_value)
{
  const gchar *raw = g_getenv (name);
  if (!raw || !*raw)
    return default_value;

  gchar *end = NULL;
  guint64 parsed = g_ascii_strtoull (raw, &end, 10);
  if (!end || *end != '\0' || parsed < min_value || parsed > G_MAXUINT)
    return default_value;
  return (guint) parsed;
}

static inline gint
read_env_int (const char *name, gint default_value, gint min_value)
{
  const gchar *raw = g_getenv (name);
  if (!raw || !*raw)
    return default_value;

  gchar *end = NULL;
  gint64 parsed = g_ascii_strtoll (raw, &end, 10);
  if (!end || *end != '\0' || parsed < min_value || parsed > G_MAXINT)
    return default_value;
  return (gint) parsed;
}

static std::string
trim_copy (const std::string &value)
{
  const size_t start = value.find_first_not_of (" \t\r\n");
  if (start == std::string::npos)
    return std::string ();
  const size_t end = value.find_last_not_of (" \t\r\n");
  return value.substr (start, end - start + 1);
}

static bool
parse_int_token (const std::string &raw, gint *out_value)
{
  if (!out_value)
    return false;

  const std::string token = trim_copy (raw);
  if (token.empty ())
    return false;

  errno = 0;
  char *end = NULL;
  long parsed = std::strtol (token.c_str (), &end, 10);
  if (errno != 0 || !end || *end != '\0')
    return false;
  if (parsed < G_MININT || parsed > G_MAXINT)
    return false;

  *out_value = (gint) parsed;
  return true;
}

static bool
parse_roi_poly (const gchar *raw, std::vector<IntrusionPoint> &out_points)
{
  out_points.clear ();
  if (!raw || !*raw)
    return true;

  std::stringstream ss (raw);
  std::string token;
  while (std::getline (ss, token, ';')) {
    const std::string point_token = trim_copy (token);
    if (point_token.empty ())
      continue;

    const size_t comma = point_token.find (',');
    if (comma == std::string::npos)
      return false;

    gint x = 0;
    gint y = 0;
    if (!parse_int_token (point_token.substr (0, comma), &x))
      return false;
    if (!parse_int_token (point_token.substr (comma + 1), &y))
      return false;

    out_points.push_back ({x, y});
  }

  return out_points.empty () || out_points.size () >= 3;
}

static void
free_source_configs (GstDsIntrusionMeta *self)
{
  if (!self->source_configs)
    return;

  delete self->source_configs;
  self->source_configs = NULL;
}

static void
close_sidecar_stream (GstDsIntrusionMeta *self)
{
  if (self->sidecar_stream) {
    self->sidecar_stream->flush ();
    delete self->sidecar_stream;
    self->sidecar_stream = NULL;
  }

  if (self->sidecar_path) {
    g_free (self->sidecar_path);
    self->sidecar_path = NULL;
  }

  self->sidecar_rows_written = 0;
}

static IntrusionSourceConfig
default_source_config (guint source_id)
{
  IntrusionSourceConfig config;
  config.label = "CH" + std::to_string (source_id);
  config.roi_status = "missing";
  return config;
}

static gboolean
load_source_configs (GstDsIntrusionMeta *self)
{
  free_source_configs (self);
  self->source_configs = new std::unordered_map<guint, IntrusionSourceConfig> ();

  self->source_count = read_env_uint ("AID_DSINTRUSIONMETA_SOURCE_COUNT",
      DEFAULT_SOURCE_COUNT, 1);
  self->person_class_id = read_env_int ("AID_DSINTRUSIONMETA_PERSON_CLASS_ID",
      DEFAULT_PERSON_CLASS_ID, 0);

  for (guint source_id = 0; source_id < self->source_count; ++source_id) {
    IntrusionSourceConfig config = default_source_config (source_id);

    gchar *label_env = g_strdup_printf ("AID_DSINTRUSIONMETA_SOURCE%u_LABEL", source_id);
    gchar *status_env = g_strdup_printf ("AID_DSINTRUSIONMETA_SOURCE%u_ROI_STATUS", source_id);
    gchar *poly_env = g_strdup_printf ("AID_DSINTRUSIONMETA_SOURCE%u_ROI_POLY", source_id);

    const gchar *label_raw = g_getenv (label_env);
    if (label_raw && *label_raw)
      config.label = label_raw;

    const gchar *status_raw = g_getenv (status_env);
    if (status_raw && *status_raw)
      config.roi_status = status_raw;

    const gchar *poly_raw = g_getenv (poly_env);
    if (!parse_roi_poly (poly_raw, config.roi_points)) {
      GST_WARNING_OBJECT (self,
          "invalid ROI polygon env for source_id=%u (%s); suppressing ROI lines",
          source_id, poly_env);
      config.roi_points.clear ();
      config.roi_status = "error";
    }

    (*self->source_configs)[source_id] = config;
    GST_INFO_OBJECT (self,
        "source_id=%u label=%s roi_status=%s roi_points=%zu",
        source_id,
        config.label.c_str (),
        config.roi_status.c_str (),
        config.roi_points.size ());

    g_free (label_env);
    g_free (status_env);
    g_free (poly_env);
  }

  return TRUE;
}

static gboolean
open_sidecar_stream (GstDsIntrusionMeta *self)
{
  close_sidecar_stream (self);

  const gchar *sidecar_raw = g_getenv ("AID_DSINTRUSIONMETA_SIDECAR_PATH");
  if (!sidecar_raw || !*sidecar_raw) {
    GST_ERROR_OBJECT (self,
        "AID_DSINTRUSIONMETA_SIDECAR_PATH is required for Stage 04.03 track export");
    return FALSE;
  }

  self->sidecar_path = g_strdup (sidecar_raw);
  self->sidecar_stream = new std::ofstream (self->sidecar_path, std::ios::out | std::ios::trunc);
  if (!self->sidecar_stream || !self->sidecar_stream->is_open ()) {
    GST_ERROR_OBJECT (self, "could not open sidecar path: %s", self->sidecar_path);
    close_sidecar_stream (self);
    return FALSE;
  }

  (*self->sidecar_stream)
      << "frame_num,source_id,track_id,mode,proxy_active,proxy_age,event,stop_reason,handoff_reason,"
      << "proxy_left,proxy_top,proxy_width,proxy_height,patch_left,patch_top,patch_width,patch_height,"
      << "patch_source,pose_anchor_source,tracked_points,flow_dx,flow_dy,flow_mag\n";
  self->sidecar_stream->flush ();
  GST_INFO_OBJECT (self, "sidecar export enabled path=%s", self->sidecar_path);
  return TRUE;
}

static void
set_text_params (NvOSD_TextParams &text, gchar *display_text, gint x, gint y,
    guint font_size, const NvOSD_ColorParams &font_color,
    const NvOSD_ColorParams &bg_color)
{
  text.display_text = display_text;
  text.x_offset = x;
  text.y_offset = std::max (0, y);
  text.set_bg_clr = 1;
  text.text_bg_clr = bg_color;
  text.font_params.font_name = (gchar *) "Serif";
  text.font_params.font_size = font_size;
  text.font_params.font_color = font_color;
}

static gboolean
point_in_polygon (const std::vector<IntrusionPoint> &polygon, gdouble x, gdouble y)
{
  if (polygon.size () < 3)
    return FALSE;

  gboolean inside = FALSE;
  for (size_t i = 0, j = polygon.size () - 1; i < polygon.size (); j = i++) {
    const gdouble xi = (gdouble) polygon[i].x;
    const gdouble yi = (gdouble) polygon[i].y;
    const gdouble xj = (gdouble) polygon[j].x;
    const gdouble yj = (gdouble) polygon[j].y;

    const gboolean intersects = ((yi > y) != (yj > y)) &&
        (x < ((xj - xi) * (y - yi) / ((yj - yi) + 1e-6)) + xi);
    if (intersects)
      inside = !inside;
  }

  return inside;
}

static gboolean
object_is_candidate (const IntrusionSourceConfig &config, const NvDsObjectMeta *obj_meta)
{
  if (config.roi_points.size () < 3 || !obj_meta)
    return FALSE;

  const gdouble rep_x = (gdouble) obj_meta->rect_params.left +
      ((gdouble) obj_meta->rect_params.width * 0.5);
  const gdouble rep_y = (gdouble) obj_meta->rect_params.top +
      (gdouble) obj_meta->rect_params.height;
  return point_in_polygon (config.roi_points, rep_x, rep_y);
}

static void
style_object_bbox (NvDsObjectMeta *obj_meta, gboolean candidate)
{
  if (!obj_meta)
    return;

  NvOSD_RectParams &rect = obj_meta->rect_params;
  NvOSD_TextParams &text = obj_meta->text_params;

  rect.border_width = candidate ? 5 : 4;
  rect.has_bg_color = 0;
  rect.border_color = candidate ?
      (NvOSD_ColorParams) {1.0, 0.62, 0.0, 1.0} :
      (NvOSD_ColorParams) {0.18, 0.90, 0.34, 1.0};

  text.set_bg_clr = 1;
  text.text_bg_clr = candidate ?
      (NvOSD_ColorParams) {0.26, 0.15, 0.0, 0.82} :
      (NvOSD_ColorParams) {0.0, 0.18, 0.05, 0.78};
  text.font_params.font_color = candidate ?
      (NvOSD_ColorParams) {1.0, 0.86, 0.70, 1.0} :
      (NvOSD_ColorParams) {0.90, 1.0, 0.90, 1.0};
}

static void
write_sidecar_row (GstDsIntrusionMeta *self, const NvDsFrameMeta *frame_meta,
    const NvDsObjectMeta *obj_meta)
{
  if (!self || !self->sidecar_stream || !frame_meta || !obj_meta)
    return;
  if (!self->sidecar_stream->is_open ())
    return;
  if (obj_meta->object_id == UNTRACKED_OBJECT_ID)
    return;

  (*self->sidecar_stream)
      << frame_meta->frame_num << ','
      << frame_meta->source_id << ','
      << static_cast<unsigned long long> (obj_meta->object_id) << ','
      << "real,0,0,tracked,,,"
      << std::fixed << std::setprecision (2)
      << obj_meta->rect_params.left << ','
      << obj_meta->rect_params.top << ','
      << obj_meta->rect_params.width << ','
      << obj_meta->rect_params.height << ','
      << "0,0,0,0,"
      << ",tracker_bbox,0,0,0,0\n";

  self->sidecar_rows_written += 1;
}

static IntrusionFrameStats
apply_object_styles_and_collect_stats (GstDsIntrusionMeta *self, NvDsFrameMeta *frame_meta,
    const IntrusionSourceConfig &config)
{
  IntrusionFrameStats stats;
  for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
    if (!obj_meta)
      continue;
    if ((gint) obj_meta->class_id != self->person_class_id)
      continue;

    stats.total_objects++;
    const gboolean is_tracked = (obj_meta->object_id != UNTRACKED_OBJECT_ID);
    if (is_tracked) {
      stats.tracked_objects++;
      write_sidecar_row (self, frame_meta, obj_meta);
    }

    const gboolean candidate = object_is_candidate (config, obj_meta);
    if (candidate) {
      stats.candidate_objects++;
      if (is_tracked)
        stats.candidate_tracked_objects++;
    }

    style_object_bbox (obj_meta, candidate);
  }

  return stats;
}

static const gchar *
frame_status_label (const IntrusionSourceConfig &config, const IntrusionFrameStats &stats)
{
  if (config.roi_status != "loaded")
    return "NO ROI";
  if (stats.candidate_objects > 0)
    return "CANDIDATE";
  return "NORMAL";
}

static void
add_monitor_display_meta (GstDsIntrusionMeta *self, NvDsBatchMeta *batch_meta,
    NvDsFrameMeta *frame_meta)
{
  IntrusionSourceConfig config = default_source_config (frame_meta->source_id);
  if (self->source_configs) {
    const auto it = self->source_configs->find (frame_meta->source_id);
    if (it != self->source_configs->end ())
      config = it->second;
  }

  const IntrusionFrameStats stats = apply_object_styles_and_collect_stats (self, frame_meta, config);
  const gchar *status_label = frame_status_label (config, stats);
  const gboolean roi_loaded = (config.roi_status == "loaded");
  const gchar *status_token = status_label;
  if (g_strcmp0 (status_label, "CANDIDATE") == 0)
    status_token = "CAND";

  gchar *status_raw = roi_loaded ?
      g_strdup_printf ("%s | ROI ON | O%u T%u H%u",
          status_token,
          stats.total_objects,
          stats.tracked_objects,
          stats.candidate_objects) :
      g_strdup_printf ("NO ROI | O%u T%u",
          stats.total_objects,
          stats.tracked_objects);
  const std::string status_line = status_raw ? status_raw : "";
  g_free (status_raw);

  const gint frame_width = self->video_info.width > 0 ? self->video_info.width : self->processing_width;
  NvDsDisplayMeta *display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
  if (!display_meta)
    return;

  display_meta->num_rects = 0;
  display_meta->num_labels = 0;
  display_meta->num_lines = 0;
  display_meta->num_arrows = 0;
  display_meta->num_circles = 0;

  const guint max_lines = 16;
  const guint max_labels = 16;

  if (config.roi_points.size () >= 3) {
    const guint line_count = std::min ((guint) config.roi_points.size (), max_lines);
    for (guint i = 0; i < line_count; ++i) {
      const IntrusionPoint &p0 = config.roi_points[i];
      const IntrusionPoint &p1 = config.roi_points[(i + 1) % config.roi_points.size ()];
      NvOSD_LineParams &line = display_meta->line_params[display_meta->num_lines++];
      line.x1 = p0.x;
      line.y1 = p0.y;
      line.x2 = p1.x;
      line.y2 = p1.y;
      line.line_width = 4;
      line.line_color = (NvOSD_ColorParams) {0.20, 0.85, 0.90, 1.0};
    }
  }

  const gint pad = 12;
  const gint left_label_x = pad;
  const gint left_label_y = 18;
  const gint right_pad = 96;
  const gint right_line_y = 18;
  const gint right_line_w = std::max (230,
      ((gint) g_utf8_strlen (status_line.c_str (), -1) * 10) + 32);
  const gint right_line_x = std::max (pad, frame_width - right_line_w - right_pad);

  if (display_meta->num_labels < max_labels) {
    NvOSD_TextParams &title = display_meta->text_params[display_meta->num_labels++];
    set_text_params (
        title,
        g_strdup (config.label.c_str ()),
        left_label_x,
        left_label_y,
        17,
        (NvOSD_ColorParams) {1.0, 1.0, 1.0, 1.0},
        (NvOSD_ColorParams) {0.0, 0.0, 0.0, 0.78});
  }

  if (display_meta->num_labels < max_labels) {
    NvOSD_TextParams &badge = display_meta->text_params[display_meta->num_labels++];
    set_text_params (
        badge,
        g_strdup (status_line.c_str ()),
        right_line_x + 14,
        right_line_y,
        11,
        (NvOSD_ColorParams) {0.92, 0.96, 0.98, 1.0},
        (NvOSD_ColorParams) {0.0, 0.0, 0.0, 0.78});
  }

  nvds_add_display_meta_to_frame (frame_meta, display_meta);
}

static void
gst_dsintrusionmeta_class_init (GstDsIntrusionMetaClass *klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstElementClass *gstelement_class = (GstElementClass *) klass;
  GstBaseTransformClass *gstbasetransform_class = (GstBaseTransformClass *) klass;

  g_setenv ("DS_NEW_BUFAPI", "1", TRUE);

  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_dsintrusionmeta_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_dsintrusionmeta_get_property);

  gstbasetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_dsintrusionmeta_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_dsintrusionmeta_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_dsintrusionmeta_stop);
  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_dsintrusionmeta_transform_ip);

  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element", 0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width", "Processing Width",
          "Compatibility property for the DeepStream ds-example hook", 1, G_MAXINT,
          DEFAULT_PROCESSING_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height", "Processing Height",
          "Compatibility property for the DeepStream ds-example hook", 1, G_MAXINT,
          DEFAULT_PROCESSING_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_FULL_FRAME,
      g_param_spec_boolean ("full-frame", "Full Frame",
          "Required for the Stage 04.03 intrusion export path", DEFAULT_PROCESS_FULL_FRAME,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_BLUR_OBJECTS,
      g_param_spec_boolean ("blur-objects", "Blur Objects",
          "Unused compatibility property", DEFAULT_BLUR_OBJECTS,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID", 0, G_MAXUINT, DEFAULT_GPU_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_BATCH_SIZE,
      g_param_spec_uint ("batch-size", "Batch Size",
          "Maximum batch size for processing", 1, NVDSEXAMPLE_MAX_BATCH_SIZE,
          DEFAULT_BATCH_SIZE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsintrusionmeta_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_dsintrusionmeta_sink_template));

  gst_element_class_set_details_simple (gstelement_class,
      "DsIntrusionMeta plugin",
      "DsIntrusionMeta Plugin",
      "Stage 04.03 intrusion metadata export plugin for DeepStream multistream tiles",
      "AID");
}

static void
gst_dsintrusionmeta_init (GstDsIntrusionMeta *self)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (self);
  gst_base_transform_set_in_place (btrans, TRUE);
  gst_base_transform_set_passthrough (btrans, TRUE);

  self->unique_id = DEFAULT_UNIQUE_ID;
  self->processing_width = DEFAULT_PROCESSING_WIDTH;
  self->processing_height = DEFAULT_PROCESSING_HEIGHT;
  self->process_full_frame = DEFAULT_PROCESS_FULL_FRAME;
  self->blur_objects = DEFAULT_BLUR_OBJECTS;
  self->gpu_id = DEFAULT_GPU_ID;
  self->max_batch_size = DEFAULT_BATCH_SIZE;
  self->source_count = DEFAULT_SOURCE_COUNT;
  self->person_class_id = DEFAULT_PERSON_CLASS_ID;
  self->source_configs = NULL;
  self->sidecar_path = NULL;
  self->sidecar_stream = NULL;
  self->sidecar_rows_written = 0;
}

static void
gst_dsintrusionmeta_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstDsIntrusionMeta *self = GST_DSINTRUSIONMETA (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      self->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      self->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      self->processing_height = g_value_get_int (value);
      break;
    case PROP_PROCESS_FULL_FRAME:
      self->process_full_frame = g_value_get_boolean (value);
      break;
    case PROP_BLUR_OBJECTS:
      self->blur_objects = g_value_get_boolean (value);
      break;
    case PROP_GPU_DEVICE_ID:
      self->gpu_id = g_value_get_uint (value);
      break;
    case PROP_BATCH_SIZE:
      self->max_batch_size = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_dsintrusionmeta_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstDsIntrusionMeta *self = GST_DSINTRUSIONMETA (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, self->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, self->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, self->processing_height);
      break;
    case PROP_PROCESS_FULL_FRAME:
      g_value_set_boolean (value, self->process_full_frame);
      break;
    case PROP_BLUR_OBJECTS:
      g_value_set_boolean (value, self->blur_objects);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, self->gpu_id);
      break;
    case PROP_BATCH_SIZE:
      g_value_set_uint (value, self->max_batch_size);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_dsintrusionmeta_start (GstBaseTransform * btrans)
{
  GstDsIntrusionMeta *self = GST_DSINTRUSIONMETA (btrans);
  if (!load_source_configs (self))
    return FALSE;
  return open_sidecar_stream (self);
}

static gboolean
gst_dsintrusionmeta_stop (GstBaseTransform * btrans)
{
  GstDsIntrusionMeta *self = GST_DSINTRUSIONMETA (btrans);
  GST_INFO_OBJECT (self, "closing sidecar rows_written=%" G_GUINT64_FORMAT, self->sidecar_rows_written);
  close_sidecar_stream (self);
  free_source_configs (self);
  return TRUE;
}

static gboolean
gst_dsintrusionmeta_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstDsIntrusionMeta *self = GST_DSINTRUSIONMETA (btrans);
  gst_video_info_from_caps (&self->video_info, incaps);
  return TRUE;
}

static GstFlowReturn
gst_dsintrusionmeta_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstDsIntrusionMeta *self = GST_DSINTRUSIONMETA (btrans);
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (inbuf);
  if (!batch_meta)
    return GST_FLOW_OK;

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
    if (!frame_meta)
      continue;
    add_monitor_display_meta (self, batch_meta, frame_meta);
  }

  if (self->sidecar_stream)
    self->sidecar_stream->flush ();
  return GST_FLOW_OK;
}

static gboolean
dsintrusionmeta_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_dsintrusionmeta_debug, "dsintrusionmeta", 0,
      "dsintrusionmeta plugin");

  if (!gst_element_register (plugin, "dsexample", GST_RANK_PRIMARY,
          GST_TYPE_DSINTRUSIONMETA))
    return FALSE;

  gst_element_register (plugin, "dsintrusionmeta", GST_RANK_NONE,
      GST_TYPE_DSINTRUSIONMETA);
  return TRUE;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_dsintrusionmeta,
    DESCRIPTION, dsintrusionmeta_plugin_init, "8.0", LICENSE, BINARY_PACKAGE, URL)
