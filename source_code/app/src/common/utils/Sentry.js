const Sentry = require('@sentry/node');

const SentryUtil = {
	init: () => {
		Sentry.init({
			dsn: process.env.SENTRY_DSN_URL,
			// We recommend adjusting this value in production, or using tracesSampler
			// for finer control
			tracesSampleRate: 1.0,
		});
		return Sentry;
	}
};

module.exports = SentryUtil.init();