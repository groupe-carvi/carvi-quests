//! Official MCP SDK (rmcp) integration.
//!
//! This module exposes the existing `McpRouter` tools via the official Rust SDK
//! (`rmcp`).
//!
//! focuses on *tools* (tool list + call_tool). Prompts
//! and resources can be mapped later.

use std::borrow::Cow;

use cquests_server::ActionService;
use futures::FutureExt;
use rmcp::handler::server::router::tool::{ToolRoute, ToolRouter};
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParam, CallToolResult, CancelledNotificationParam, CompleteRequestParam,
    CompleteResult, CustomRequest, CustomResult, GetPromptRequestParam, GetPromptResult,
    Implementation, InitializeRequestParam, InitializeResult, JsonObject, ListPromptsResult,
    ListResourceTemplatesResult, ListResourcesResult, PaginatedRequestParam, ProgressNotificationParam,
    Prompt, ReadResourceRequestParam, ReadResourceResult, Resource, ResourceTemplate,
    ServerCapabilities, ServerInfo, SetLevelRequestParam, SubscribeRequestParam, Tool,
    UnsubscribeRequestParam,
};
use rmcp::service::{RoleClient, RoleServer, RunningService, ServiceExt};
use rmcp::service::{NotificationContext, RequestContext};
use rmcp::ErrorData;
use rmcp::transport::async_rw::AsyncRwTransport;
use serde_json::Value;

use crate::{auth_gm, McpError, McpRouter};

/// Server-side handler implementing rmcp's `ServerHandler`.
///
/// Tools are executed by proxying into the existing in-process `McpRouter`.
///
/// Note: today this assumes GM authority for tool execution.
#[derive(Clone)]
pub struct CquestsSdkHandler<S: ActionService> {
    inner: McpRouter<S>,
    pub tool_router: ToolRouter<Self>,
    info: ServerInfo,
}

impl<S> CquestsSdkHandler<S>
where
    S: ActionService + Send + Sync + 'static,
{
    pub fn new(service: S) -> Self {
        let inner = McpRouter::new(service);

        let mut tool_router = ToolRouter::new();
        tool_router.add_route(tool_route::<S>(
            crate::TOOL_GET_VISIBLE_STATE,
            "Get per-player filtered visible state",
            rmcp::object!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "player_id": {"type": "integer"}
                },
                "required": ["session_id", "player_id"]
            }),
        ));
        tool_router.add_route(tool_route::<S>(
            crate::TOOL_ROLL,
            "Roll dice using server-owned deterministic RNG",
            rmcp::object!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "dice_expr": {"type": "string"}
                },
                "required": ["session_id", "dice_expr"]
            }),
        ));
        tool_router.add_route(tool_route::<S>(
            crate::TOOL_MOVE,
            "Move player to an adjacent destination",
            rmcp::object!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "player_id": {"type": "integer"},
                    "destination_id": {"type": "integer"}
                },
                "required": ["session_id", "player_id", "destination_id"]
            }),
        ));
        tool_router.add_route(tool_route::<S>(
            crate::TOOL_INSPECT,
            "Inspect an entity or your current location",
            rmcp::object!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "player_id": {"type": "integer"},
                    "target": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string"},
                            "id": {"type": "integer"}
                        }
                    }
                },
                "required": ["session_id", "player_id", "target"]
            }),
        ));
        tool_router.add_route(tool_route::<S>(
            crate::TOOL_ATTACK,
            "Attack a target entity (co-located)",
            rmcp::object!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "player_id": {"type": "integer"},
                    "target_id": {"type": "integer"}
                },
                "required": ["session_id", "player_id", "target_id"]
            }),
        ));
        tool_router.add_route(tool_route::<S>(
            crate::TOOL_SEND_TO_PLAYERS,
            "Send an out-of-band text message to players (communication only; no game-state change).",
            rmcp::object!({
                "type": "object",
                "properties": {
                    "session_id": {"type": "integer"},
                    "to_player_ids": {"type": "array", "items": {"type": "integer"}},
                    "text": {"type": "string"}
                },
                "required": ["session_id", "text"]
            }),
        ));

        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = Implementation {
            name: "carvi-quests".to_string(),
            title: Some("Carvi Quests".to_string()),
            version: env!("CARGO_PKG_VERSION").to_string(),
            icons: None,
            website_url: None,
        };
        info.instructions = Some(
            "CQuests MCP server (rmcp SDK). Tools are available; prompts/resources are not implemented yet."
                .to_string(),
        );

        Self {
            inner,
            tool_router,
            info,
        }
    }
}

fn tool_route<S>(name: &'static str, description: &'static str, input_schema: JsonObject) -> ToolRoute<CquestsSdkHandler<S>>
where
    S: ActionService + Send + Sync + 'static,
{
    let tool = Tool::new(name, description, input_schema);

    ToolRoute::new_dyn(tool, move |ctx: ToolCallContext<'_, CquestsSdkHandler<S>>| {
        async move {
            let auth = auth_gm();
            let args_obj: JsonObject = ctx.arguments.clone().unwrap_or_default();
            let args_val = Value::Object(args_obj);

            let out = match ctx.service.inner.call_tool(&auth, ctx.name.as_ref(), args_val) {
                Ok(v) => CallToolResult::structured(v),
                Err(e) => CallToolResult::structured_error(mcp_error_to_json(e, ctx.name.as_ref())),
            };

            Ok(out)
        }
        .boxed()
    })
}

fn mcp_error_to_json(err: McpError, tool: &str) -> Value {
    serde_json::json!({
        "error": "tool_call_failed",
        "tool": tool,
        "details": err.to_string(),
    })
}

/// Auto-generate `call_tool` + `list_tools` based on `self.tool_router`.
#[rmcp::tool_handler]
impl<S> ServerHandler for CquestsSdkHandler<S>
where
    S: ActionService + Send + Sync + 'static,
{
    fn ping(&self, _context: RequestContext<RoleServer>) -> impl std::future::Future<Output = Result<(), ErrorData>> + Send + '_ {
        async { Ok(()) }
    }

    fn initialize(
        &self,
        _request: InitializeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<InitializeResult, ErrorData>> + Send + '_ {
        async { Ok(self.info.clone()) }
    }

    fn complete(
        &self,
        _request: CompleteRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CompleteResult, ErrorData>> + Send + '_ {
        async {
            Err(ErrorData::internal_error(
                "completions are not implemented by this server",
                None,
            ))
        }
    }

    fn set_level(
        &self,
        _request: SetLevelRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<(), ErrorData>> + Send + '_ {
        async { Ok(()) }
    }

    fn get_prompt(
        &self,
        _request: GetPromptRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<GetPromptResult, ErrorData>> + Send + '_ {
        async {
            Err(ErrorData::internal_error(
                "prompts are not implemented by this server",
                None,
            ))
        }
    }

    fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListPromptsResult, ErrorData>> + Send + '_ {
        async { Ok(ListPromptsResult::with_all_items(Vec::<Prompt>::new())) }
    }

    fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourcesResult, ErrorData>> + Send + '_ {
        async { Ok(ListResourcesResult::with_all_items(Vec::<Resource>::new())) }
    }

    fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourceTemplatesResult, ErrorData>> + Send + '_ {
        async {
            Ok(ListResourceTemplatesResult::with_all_items(
                Vec::<ResourceTemplate>::new(),
            ))
        }
    }

    fn read_resource(
        &self,
        _request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ReadResourceResult, ErrorData>> + Send + '_ {
        async {
            Err(ErrorData::internal_error(
                "resources are not implemented by this server",
                None,
            ))
        }
    }

    fn subscribe(
        &self,
        _request: SubscribeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<(), ErrorData>> + Send + '_ {
        async { Ok(()) }
    }

    fn unsubscribe(
        &self,
        _request: UnsubscribeRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<(), ErrorData>> + Send + '_ {
        async { Ok(()) }
    }

    fn on_custom_request(
        &self,
        _request: CustomRequest,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CustomResult, ErrorData>> + Send + '_ {
        async { Err(ErrorData::internal_error("custom requests not supported", None)) }
    }

    fn on_cancelled(
        &self,
        _notification: CancelledNotificationParam,
        _context: NotificationContext<RoleServer>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {}
    }

    fn on_progress(
        &self,
        _notification: ProgressNotificationParam,
        _context: NotificationContext<RoleServer>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {}
    }

    fn on_initialized(
        &self,
        _context: NotificationContext<RoleServer>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {}
    }

    fn on_roots_list_changed(
        &self,
        _context: NotificationContext<RoleServer>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {}
    }

    fn on_custom_notification(
        &self,
        _notification: rmcp::model::CustomNotification,
        _context: NotificationContext<RoleServer>,
    ) -> impl std::future::Future<Output = ()> + Send + '_ {
        async {}
    }

    fn get_info(&self) -> ServerInfo {
        self.info.clone()
    }
}

/// A small async wrapper around an rmcp running client that preserves the
/// project’s existing “tool name + JSON args -> JSON result” calling style.
///
/// Note: This client returns the tool’s `structured_content` JSON value.
pub struct SdkClient {
    running: RunningService<RoleClient, ()>,
}

impl SdkClient {
    pub fn new(running: RunningService<RoleClient, ()>) -> Self {
        Self { running }
    }

    pub async fn list_all_tools(&self) -> Result<Vec<Tool>, rmcp::service::ServiceError> {
        self.running.peer().list_all_tools().await
    }

    pub async fn call_tool_json(
        &self,
        name: impl Into<Cow<'static, str>>,
        arguments: Value,
    ) -> Result<Value, rmcp::service::ServiceError> {
        let arguments = match arguments {
            Value::Null => None,
            Value::Object(map) => Some(map),
            other => Some(rmcp::object!({ "value": other })),
        };

        let res = self
            .running
            .peer()
            .call_tool(CallToolRequestParam {
                name: name.into(),
                arguments,
                task: None,
            })
            .await?;

        Ok(res.structured_content.unwrap_or(Value::Null))
    }
}

/// Convenience helper for tests and in-process usage: connect an rmcp client to
/// an rmcp server over an in-memory duplex stream.
pub async fn connect_inprocess<S>(service: S) -> Result<SdkClient, rmcp::service::ClientInitializeError>
where
    S: ActionService + Send + Sync + 'static,
{
    let server = CquestsSdkHandler::new(service);

    let (a, b) = tokio::io::duplex(64 * 1024);
    let (a_r, a_w) = tokio::io::split(a);
    let (b_r, b_w) = tokio::io::split(b);

    let server_transport = AsyncRwTransport::new_server(b_r, b_w);
    let client_transport = AsyncRwTransport::new_client(a_r, a_w);

    // Start the server first so it can accept the client init.
    let _server_task = tokio::spawn(async move {
        let _running: RunningService<RoleServer, _> = server.serve(server_transport).await?;
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok::<(), rmcp::service::ServerInitializeError>(())
    });

    let running_client: RunningService<RoleClient, ()> = ().serve(client_transport).await?;
    Ok(SdkClient::new(running_client))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cquests_core::PlayerId;
    use cquests_server::InMemoryActionService;

    #[tokio::test]
    async fn rmcp_roundtrip_get_visible_state() {
        let service = InMemoryActionService::new();
        let session_id = service.create_session(123, InMemoryActionService::default_demo_world(PlayerId::new(1)));

        let client = connect_inprocess(service).await.expect("connect inprocess");

        let tools = client.list_all_tools().await.expect("list tools");
        assert!(tools.iter().any(|t| t.name == crate::TOOL_GET_VISIBLE_STATE));

        let out = client
            .call_tool_json(
                crate::TOOL_GET_VISIBLE_STATE,
                serde_json::json!({"session_id": session_id, "player_id": 1}),
            )
            .await
            .expect("call tool");

        let visible: cquests_core::VisibleState = serde_json::from_value(out).expect("visible state json");
        assert_eq!(visible.player_id, PlayerId::new(1));
    }
}
