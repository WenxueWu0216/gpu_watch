{{- define "gpuwatch.labels" -}}
app.kubernetes.io/part-of: gpuwatch
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
