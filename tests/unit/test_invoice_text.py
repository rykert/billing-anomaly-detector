from billing_anomaly_detector.application.services.invoice_text import invoice_to_text

from tests.conftest import make_invoice


class TestInvoiceToText:
    def test_contains_member_id(self) -> None:
        invoice = make_invoice(member_id="MBR999")
        text = invoice_to_text(invoice)
        assert "MBR999" in text

    def test_contains_claim_code(self) -> None:
        invoice = make_invoice(claim_code="99213")
        text = invoice_to_text(invoice)
        assert "99213" in text

    def test_contains_ratio_to_three_decimal_places(self) -> None:
        invoice = make_invoice(billed_amount="150.00", allowed_amount="100.00")
        text = invoice_to_text(invoice)
        assert "1.500" in text  # 150/100 = 1.5 formatted to 3dp

    def test_contains_provider_npi(self) -> None:
        invoice = make_invoice(provider_npi="9876543210")
        text = invoice_to_text(invoice)
        assert "9876543210" in text

    def test_returns_string(self) -> None:
        invoice = make_invoice()
        assert isinstance(invoice_to_text(invoice), str)

    def test_contains_service_date(self) -> None:
        from datetime import date
        invoice = make_invoice(service_date=date(2009, 6, 15))
        text = invoice_to_text(invoice)
        assert "2009-06-15" in text
