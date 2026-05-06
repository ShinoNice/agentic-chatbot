terraform {
  backend "azurerm" {
    # Configure via -backend-config flags or .tfbackend file at init time:
    #   resource_group_name  = "rg-tfstate-agentic-chatbot"
    #   storage_account_name = "sttfstateagenticchatbot"
    #   container_name       = "tfstate"
    #   key                  = "agentic-chatbot.tfstate"
  }
}
