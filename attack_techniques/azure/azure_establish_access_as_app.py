import json
import subprocess
from typing import Any, Dict, Tuple

from core.azure.azure_access import AzureAccess

from ..base_technique import (
    AzureTRMTechnique,
    BaseTechnique,
    ExecutionStatus,
    MitreTechnique,
)
from ..technique_registry import TechniqueRegistry


@TechniqueRegistry.register
class AzureEstablishAccessAsApp(BaseTechnique):
    def __init__(self):
        mitre_techniques = [
            MitreTechnique(
                technique_id="T1098.003",
                technique_name="Valid Accounts",
                tactics=[
                    "Defense Evasion",
                    "Persistence",
                    "Privilege Escalation",
                    "Initial Access",
                ],
                sub_technique_name="Cloud Accounts",
            )
        ]
        azure_trm_technique = [
            AzureTRMTechnique(
                technique_id="AZT201.2",
                technique_name="Valid Credentials",
                tactics=["Initial Access"],
                sub_technique_name="Service Principal",
            )
        ]
        super().__init__(
            "Establish Access As App",
            "Authenticates to an Azure tenant using service principal credentials (client ID and client secret) to establish programmatic access. The technique can attempt authentication with or without subscription access and returns information about accessible subscriptions. Successfully authenticated service principal credentials can be used for subsequent automated interactions with Azure resources.",
            mitre_techniques,
            azure_trm_technique,
        )

    def execute(self, **kwargs: Any) -> Tuple[ExecutionStatus, Dict[str, Any]]:
        self.validate_parameters(kwargs)
        try:
            # input validation
            app_id: str = kwargs["app_id"]
            app_secret: str = kwargs["app_secret"]
            tenant_id: str = kwargs["tenant_id"]
            allow_no_sub_login: str = kwargs.get("allow_no_sub_login", True)

            # get az full execution path
            az_command = AzureAccess().az_command

            if not allow_no_sub_login:
                raw_response = subprocess.run(
                    [
                        az_command,
                        "login",
                        "--service-principal",
                        "-u",
                        app_id,
                        f"-p={app_secret}",
                        "--tenant",
                        tenant_id,
                    ],
                    capture_output=True,
                )
            else:
                raw_response = subprocess.run(
                    [
                        az_command,
                        "login",
                        "--service-principal",
                        "-u",
                        app_id,
                        f"-p={app_secret}",
                        "--tenant",
                        tenant_id,
                        "--allow-no-subscriptions",
                    ],
                    capture_output=True,
                )

            output = raw_response.stdout
            struc_output = json.loads(output.decode("utf-8"))

            if raw_response.returncode == 0:
                try:
                    output = {}
                    for subscription in struc_output:
                        output[subscription["id"]] = {
                            "subscription_name": subscription.get("name", "N/A"),
                            "subscription_id": subscription.get("id", "N/A"),
                            "home_tenant_id": subscription.get("homeTenantId", "N/A"),
                            "state": subscription.get("state", "N/A"),
                            "identity": subscription.get("user", "N/A").get(
                                "name", "N/A"
                            ),
                            "identity_type": subscription.get("user", "N/A").get(
                                "type", "N/A"
                            ),
                        }
                    return ExecutionStatus.SUCCESS, {
                        "message": "Successfully established access to target Azure tenant",
                        "value": output,
                    }
                except:
                    return ExecutionStatus.PARTIAL_SUCCESS, {
                        "message": "Successfully established access to target Azure tenant",
                        "value": struc_output,
                    }
            else:
                return ExecutionStatus.FAILURE, {
                    "error": str(raw_response.returncode),
                    "message": "Failed to establish access to Azure tenant",
                }
        except Exception as e:
            return ExecutionStatus.FAILURE, {
                "error": str(e),
                "message": "Failed to establish access to Azure tenant",
            }

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "app_id": {
                "type": "str",
                "required": True,
                "default": None,
                "name": "App ID",
                "input_field_type": "text",
            },
            "app_secret": {
                "type": "str",
                "required": True,
                "default": None,
                "name": "App Secret",
                "input_field_type": "password",
            },
            "tenant_id": {
                "type": "str",
                "required": True,
                "default": None,
                "name": "Tenant ID",
                "input_field_type": "text",
            },
            "allow_no_sub_login": {
                "type": "bool",
                "required": False,
                "default": True,
                "name": "Attempt Login Without Subscription",
                "input_field_type": "bool",
            },
        }
