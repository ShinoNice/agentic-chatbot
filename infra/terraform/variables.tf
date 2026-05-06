variable "location" {
  description = "Azure region for the ACA environment (North Europe has free-tier quota)."
  type        = string
  default     = "northeurope"
}

variable "name_prefix" {
  description = "Short-name prefix for resources (<= 11 chars, lowercase)."
  type        = string
  default     = "chatbot"
}

variable "resource_group_name" {
  description = "Name of the resource group hosting the stack (must already exist)."
  type        = string
  default     = "rg-agentic-chatbot"
}

variable "acr_name" {
  description = "Name of the existing Azure Container Registry (assumed to be in this RG)."
  type        = string
  default     = "acrhybridchatbot"
}

variable "key_vault_name" {
  description = "Name of the existing Key Vault that stores LLM/Pinecone/LangSmith secrets (assumed to be in this RG)."
  type        = string
  default     = "kv-hybrid-chatbot"
}

variable "image_tag" {
  description = "Image tag to deploy for both apps (e.g. \"v5\")."
  type        = string
  default     = "v5"
}

variable "cpu" {
  description = "CPU cores per replica (0.25-2.0 in increments of 0.25)."
  type        = string
  default     = "1.0"
}

variable "memory" {
  description = "Memory per replica (e.g. 1Gi, 2Gi)."
  type        = string
  default     = "2.0Gi"
}

variable "langsmith_project" {
  description = "LangSmith project name for tracing; leave empty to disable."
  type        = string
  default     = "Agentic_Chatbot"
}

variable "cors_extra_origins" {
  description = "Public FQDN(s) allowed by the API CORS policy (in addition to localhost)."
  type        = list(string)
  default     = []
}
