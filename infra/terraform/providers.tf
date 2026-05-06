provider "azurerm" {
  features {}
  use_oidc = true # GitHub Actions auth
  # subscription_id, tenant_id, client_id come from env vars
  #   ARM_SUBSCRIPTION_ID, ARM_TENANT_ID, ARM_CLIENT_ID
}
