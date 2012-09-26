"""
oauthlib.oauth2_draft28.parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains methods related to `Section 4`_ of the OAuth 2 draft.

.. _`Section 4`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-4
"""

import json
import urlparse
from oauthlib.common import add_params_to_uri, add_params_to_qs


def prepare_grant_uri(uri, client_id, response_type, redirect_uri=None,
            scope=None, state=None, **kwargs):
    """Prepare the authorization grant request URI.

    The client constructs the request URI by adding the following
    parameters to the query component of the authorization endpoint URI
    using the "application/x-www-form-urlencoded" format as defined by
    [W3C.REC-html401-19991224]:

    response_type
            REQUIRED.  Value MUST be set to "code".
    client_id
            REQUIRED.  The client identifier as described in `Section 2.2`_.
    redirect_uri
            OPTIONAL.  As described in `Section 3.1.2`_.
    scope
            OPTIONAL.  The scope of the access request as described by
            `Section 3.3`_.
    state
            RECOMMENDED.  An opaque value used by the client to maintain
            state between the request and callback.  The authorization
            server includes this value when redirecting the user-agent back
            to the client.  The parameter SHOULD be used for preventing
            cross-site request forgery as described in `Section 10.12`_.

    GET /authorize?response_type=code&client_id=s6BhdRkqt3&state=xyz
        &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb HTTP/1.1
    Host: server.example.com

    .. _`W3C.REC-html401-19991224`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#ref-W3C.REC-html401-19991224
    .. _`Section 2.2`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-2.2
    .. _`Section 3.1.2`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-3.1.2
    .. _`Section 3.3`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-3.3
    .. _`section 10.12`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-10.12
    """
    params = [((u'response_type', response_type)),
              ((u'client_id', client_id))]

    if redirect_uri:
        params.append((u'redirect_uri', redirect_uri))
    if scope:
        params.append((u'scope', scope))
    if state:
        params.append((u'state', state))

    for k in kwargs:
        params.append((unicode(k), kwargs[k]))

    return add_params_to_uri(uri, params)


def prepare_token_request(grant_type, body=u'', **kwargs):
    """Prepare the access token request.

    The client makes a request to the token endpoint by adding the
    following parameters using the "application/x-www-form-urlencoded"
    format in the HTTP request entity-body:

    grant_type
            REQUIRED.  Value MUST be set to "authorization_code".
    code
            REQUIRED.  The authorization code received from the
            authorization server.
    redirect_uri
            REQUIRED, if the "redirect_uri" parameter was included in the
            authorization request as described in `Section 4.1.1`_, and their
            values MUST be identical.

    grant_type=authorization_code&code=SplxlOBeZQQYbYS6WxSbIA
    &redirect_uri=https%3A%2F%2Fclient%2Eexample%2Ecom%2Fcb

    .. _`Section 4.1.1`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-4.1.1
    """
    params = [(u'grant_type', grant_type)]
    for k in kwargs:
        params.append((unicode(k), kwargs[k]))

    return add_params_to_qs(body, params)


def parse_authorization_code_response(uri, state=None):
    """Parse authorization grant response URI into a dict.

    If the resource owner grants the access request, the authorization
    server issues an authorization code and delivers it to the client by
    adding the following parameters to the query component of the
    redirection URI using the "application/x-www-form-urlencoded" format:

    code
            REQUIRED.  The authorization code generated by the
            authorization server.  The authorization code MUST expire
            shortly after it is issued to mitigate the risk of leaks.  A
            maximum authorization code lifetime of 10 minutes is
            RECOMMENDED.  The client MUST NOT use the authorization code
            more than once.  If an authorization code is used more than
            once, the authorization server MUST deny the request and SHOULD
            revoke (when possible) all tokens previously issued based on
            that authorization code.  The authorization code is bound to
            the client identifier and redirection URI.
    state
            REQUIRED if the "state" parameter was present in the client
            authorization request.  The exact value received from the
            client.

    For example, the authorization server redirects the user-agent by
    sending the following HTTP response:

    HTTP/1.1 302 Found
    Location: https://client.example.com/cb?code=SplxlOBeZQQYbYS6WxSbIA
            &state=xyz

    """
    query = urlparse.urlparse(uri).query
    params = dict(urlparse.parse_qsl(query))

    if not u'code' in params:
        raise KeyError("Missing code parameter in response.")

    if state and params.get(u'state', None) != state:
        raise ValueError("Mismatching or missing state in response.")

    return params


def parse_implicit_response(uri, state=None, scope=None):
    """Parse the implicit token response URI into a dict.

    If the resource owner grants the access request, the authorization
    server issues an access token and delivers it to the client by adding
    the following parameters to the fragment component of the redirection
    URI using the "application/x-www-form-urlencoded" format:

    access_token
            REQUIRED.  The access token issued by the authorization server.
    token_type
            REQUIRED.  The type of the token issued as described in
            Section 7.1.  Value is case insensitive.
    expires_in
            RECOMMENDED.  The lifetime in seconds of the access token.  For
            example, the value "3600" denotes that the access token will
            expire in one hour from the time the response was generated.
            If omitted, the authorization server SHOULD provide the
            expiration time via other means or document the default value.
    scope
            OPTIONAL, if identical to the scope requested by the client,
            otherwise REQUIRED.  The scope of the access token as described
            by Section 3.3.
    state
            REQUIRED if the "state" parameter was present in the client
            authorization request.  The exact value received from the
            client.

    HTTP/1.1 302 Found
    Location: http://example.com/cb#access_token=2YotnFZFEjr1zCsicMWpAA
            &state=xyz&token_type=example&expires_in=3600
    """
    fragment = urlparse.urlparse(uri).fragment
    params = dict(urlparse.parse_qsl(fragment, keep_blank_values=True))
    validate_token_parameters(params, scope)

    if state and params.get(u'state', None) != state:
        raise ValueError("Mismatching or missing state in params.")

    return params


def parse_token_response(body, scope=None):
    """Parse the JSON token response body into a dict.

    The authorization server issues an access token and optional refresh
    token, and constructs the response by adding the following parameters
    to the entity body of the HTTP response with a 200 (OK) status code:

    access_token
            REQUIRED.  The access token issued by the authorization server.
    token_type
            REQUIRED.  The type of the token issued as described in
            `Section 7.1`_.  Value is case insensitive.
    expires_in
            RECOMMENDED.  The lifetime in seconds of the access token.  For
            example, the value "3600" denotes that the access token will
            expire in one hour from the time the response was generated.
            If omitted, the authorization server SHOULD provide the
            expiration time via other means or document the default value.
    refresh_token
            OPTIONAL.  The refresh token which can be used to obtain new
            access tokens using the same authorization grant as described
            in `Section 6`_.
    scope
            OPTIONAL, if identical to the scope requested by the client,
            otherwise REQUIRED.  The scope of the access token as described
            by `Section 3.3`_.

    The parameters are included in the entity body of the HTTP response
    using the "application/json" media type as defined by [`RFC4627`_].  The
    parameters are serialized into a JSON structure by adding each
    parameter at the highest structure level.  Parameter names and string
    values are included as JSON strings.  Numerical values are included
    as JSON numbers.  The order of parameters does not matter and can
    vary.

    For example:

        HTTP/1.1 200 OK
        Content-Type: application/json;charset=UTF-8
        Cache-Control: no-store
        Pragma: no-cache

        {
        "access_token":"2YotnFZFEjr1zCsicMWpAA",
        "token_type":"example",
        "expires_in":3600,
        "refresh_token":"tGzv3JOkF0XG5Qx2TlKWIA",
        "example_parameter":"example_value"
        }

    .. _`Section 7.1`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-7.1
    .. _`Section 6`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-6
    .. _`Section 3.3`: http://tools.ietf.org/html/draft-ietf-oauth-v2-28#section-3.3
    .. _`RFC4627`: http://tools.ietf.org/html/rfc4627
    """
    params = json.loads(body)
    validate_token_parameters(params, scope)
    return params


def validate_token_parameters(params, scope=None):
    """Ensures token precence, token type, expiration and scope in params."""

    if not u'access_token' in params:
        raise KeyError("Missing access token parameter.")

    if not u'token_type' in params:
        raise KeyError("Missing token type parameter.")

    # If the issued access token scope is different from the one requested by
    # the client, the authorization server MUST include the "scope" response
    # parameter to inform the client of the actual scope granted.
    # http://tools.ietf.org/html/draft-ietf-oauth-v2-25#section-3.3
    new_scope = params.get(u'scope', None)
    if scope and new_scope and scope != new_scope:
        raise Warning("Scope has changed to %s." % new_scope)
