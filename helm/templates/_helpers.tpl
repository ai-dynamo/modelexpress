# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{{/*
Expand the name of the chart.
*/}}
{{- define "modelexpress.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "modelexpress.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "modelexpress.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "modelexpress.labels" -}}
helm.sh/chart: {{ include "modelexpress.chart" . }}
{{ include "modelexpress.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "modelexpress.selectorLabels" -}}
app.kubernetes.io/name: {{ include "modelexpress.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Convert a simple Prometheus duration to seconds, for comparison.

Handles the single-unit forms these values realistically take (30s, 1m, 1h) and
returns 0 for anything else, which callers treat as "cannot check" rather than
as zero -- guessing at an exotic duration would be worse than not checking.
*/}}
{{- define "modelexpress.durationSeconds" -}}
{{- $d := . | toString -}}
{{- if regexMatch "^[0-9]+s$" $d -}}
{{- regexReplaceAll "s$" $d "" | atoi -}}
{{- else if regexMatch "^[0-9]+m$" $d -}}
{{- mul (regexReplaceAll "m$" $d "" | atoi) 60 -}}
{{- else if regexMatch "^[0-9]+h$" $d -}}
{{- mul (regexReplaceAll "h$" $d "" | atoi) 3600 -}}
{{- else -}}
0
{{- end -}}
{{- end }}

{{/*
Reject a scrapeTimeout longer than its interval.

The Prometheus Operator refuses such an endpoint with "scrapeTimeout greater
than scrapeInterval" and drops the whole PodMonitor. It reports that only as a
Kubernetes Warning event on the object -- which many clusters forbid it from
writing -- so the usual symptom is simply no targets, with nothing to read.
Since scrapeTimeout defaults to 10s here, lowering interval alone is enough to
trigger it. Takes (interval, scrapeTimeout, values-path-for-the-message).
*/}}
{{- define "modelexpress.checkScrapeTimeout" -}}
{{- $interval := include "modelexpress.durationSeconds" (index . 0) | int -}}
{{- $timeout := include "modelexpress.durationSeconds" (index . 1) | int -}}
{{- $path := index . 2 -}}
{{- if and (gt $interval 0) (gt $timeout 0) (gt $timeout $interval) -}}
{{- fail (printf "%s.scrapeTimeout (%v) is longer than %s.interval (%v): the Prometheus Operator rejects that endpoint and drops the PodMonitor, usually with no visible error. Lower scrapeTimeout to at most the interval." $path (index . 1) $path (index . 0)) -}}
{{- end -}}
{{- end }}

{{/*
Look up an alert threshold, honouring an explicit zero.

sprig's `default` treats 0 as empty, so `$t.x | default 0.05` silently replaces a
threshold deliberately set to 0 -- turning "alert on any occurrence" into the
built-in rate. Takes (thresholds-map, key, fallback).
*/}}
{{- define "modelexpress.threshold" -}}
{{- $thresholds := index . 0 -}}
{{- $key := index . 1 -}}
{{- $fallback := index . 2 -}}
{{- if hasKey $thresholds $key }}{{ index $thresholds $key }}{{ else }}{{ $fallback }}{{ end -}}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "modelexpress.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "modelexpress.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
