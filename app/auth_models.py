from __future__ import annotations

from pydantic import BaseModel, Field


class UserPublic(BaseModel):
    user_id: str
    email: str
    name: str | None = None
    picture_url: str | None = None
    is_admin: bool = False


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")
    user: UserPublic


class GoogleIdTokenRequest(BaseModel):
    id_token: str | None = Field(default=None, description="Google Sign-In ID token")
    credential: str | None = Field(default=None, description="Alias for id_token (GIS button payload)")


class EmailRegisterRequest(BaseModel):
    email: str = Field(description="Account email (not verified at signup)")
    password: str = Field(min_length=8, description="Password (min 8 characters)")
    name: str | None = Field(default=None, description="Optional display name")


class EmailLoginRequest(BaseModel):
    email: str
    password: str = Field(min_length=1)


class AuthMessageResponse(BaseModel):
    message: str
