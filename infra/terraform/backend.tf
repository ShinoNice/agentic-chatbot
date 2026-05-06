terraform {
  backend "azurerm" {
    # AAD-only auth: skip the Microsoft.Storage/storageAccounts/listKeys
    # call so the SP doesn't need the Storage Account Contributor role.
    # Only requires:
    #   - Reader on the SA (or RG)
    #   - Storage Blob Data Contributor on the SA (or container)
    use_azuread_auth = true

    # Other args configured via -backend-config flags / .tfbackend file at
    # init time:
    #   resource_group_name  = "rg-tfstate-agentic-chatbot"
    #   storage_account_name = "sttfstateagenticchatbot"
    #   container_name       = "tfstate"
    #   key                  = "agentic-chatbot.tfstate"
  }
}
