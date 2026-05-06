output "ui_fqdn" {
  description = "Public FQDN of the UI container app's ingress."
  value       = azurerm_container_app.ui.ingress[0].fqdn
}

output "ui_public_url" {
  description = "Public HTTPS URL of the UI container app."
  value       = "https://${azurerm_container_app.ui.ingress[0].fqdn}"
}

output "api_internal_url" {
  description = "Internal DNS the UI uses to reach the API inside the ACA env."
  value       = "http://ca-${var.name_prefix}-api"
}

output "managed_identity_id" {
  description = "Resource ID of the user-assigned managed identity shared by both apps."
  value       = azurerm_user_assigned_identity.uami.id
}

output "cors_hint" {
  description = "Suggested CORS origins list for the API settings.yaml."
  value = concat(
    [
      "https://${azurerm_container_app.ui.ingress[0].fqdn}",
      "http://localhost:8501",
      "http://localhost:8001",
    ],
    var.cors_extra_origins,
  )
}
