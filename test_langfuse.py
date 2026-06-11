from app.config.settings import settings

print("=== VERIFICANDO CONFIGURAÇÕES ===")
print(f"Public Key: {settings.langfuse_public_key}")
print(f"Secret Key: {settings.langfuse_secret_key[:15] if settings.langfuse_secret_key else 'VAZIA'}")
print(f"Host: {settings.langfuse_host}")
print(f"LangFuse enabled: {settings.langfuse_enabled}")

if settings.langfuse_enabled:
    from langfuse import Langfuse
    lf = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    result = lf.auth_check()
    print(f"Auth check: {result}")
else:
    print("❌ LangFuse não está habilitado — chaves não encontradas no .env")