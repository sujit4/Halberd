from typing import Any, Dict, Tuple

import boto3
from botocore.exceptions import ClientError

from ..base_technique import BaseTechnique, ExecutionStatus, MitreTechnique
from ..technique_registry import TechniqueRegistry


@TechniqueRegistry.register
class AWSEnumerateIAMUsers(BaseTechnique):
    def __init__(self):
        mitre_techniques = [
            MitreTechnique(
                technique_id="T1087.004",
                technique_name="Account Discovery",
                tactics=["Discovery"],
                sub_technique_name="Cloud Account",
            )
        ]
        super().__init__(
            "Enumerate IAM Users",
            "Enumerates all IAM users. Optionally, supply path prefix to enumerate specific users",
            mitre_techniques,
        )

    def execute(self, **kwargs: Any) -> Tuple[ExecutionStatus, Dict[str, Any]]:
        self.validate_parameters(kwargs)
        try:
            path_prefix: str = kwargs.get("path_prefix", None)

            # Ref: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/iam/client/list_users.html

            # Initialize boto3 client
            my_client = boto3.client("iam")

            if path_prefix in [None, ""]:
                # list all iam roles
                raw_response = my_client.list_users()
            else:
                # list roles with supplied path prefix
                raw_response = my_client.list_users(PathPrefix=path_prefix)

            if 200 <= raw_response["ResponseMetadata"]["HTTPStatusCode"] < 300:
                # Create output
                users = [user["UserName"] for user in raw_response["Users"]]

                return ExecutionStatus.SUCCESS, {
                    "message": f"Successfully enumerated {len(users)} users"
                    if users
                    else "No users found",
                    "value": users,
                }

            return ExecutionStatus.FAILURE, {
                "error": raw_response.get("ResponseMetadata", "N/A"),
                "message": "Failed to enumerate IAM users",
            }

        except ClientError as e:
            return ExecutionStatus.FAILURE, {
                "error": str(e),
                "message": "Failed to enumerate IAM users",
            }
        except Exception as e:
            return ExecutionStatus.FAILURE, {
                "error": str(e),
                "message": "Failed to enumerate IAM users",
            }

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "path_prefix": {
                "type": "str",
                "required": False,
                "default": None,
                "name": "Role Path Prefix",
                "input_field_type": "text",
            }
        }
